import math

import torch
import torch.nn.functional as F


def _masked_mean(values, no_model_batch):
    mask = (no_model_batch["label"] != -100).float()
    mask_sum = mask.sum()
    masked_sum = (values * mask).sum()
    return masked_sum / mask_sum if mask_sum > 0 else masked_sum


def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """Calculate D^(alpha, beta) divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)
    elif abs(alpha) < eps:
        safe_log_ratio = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(
            torch.exp(beta * log_q) * (beta * safe_log_ratio - 1) + torch.exp(beta * log_p),
            dim=-1,
        ) / (beta ** 2)
    elif abs(beta) < eps:
        safe_log_ratio = torch.where(torch.isfinite(log_p - log_q), log_p - log_q, 0.0)
        divergence = torch.sum(
            torch.exp(alpha * log_p) * (alpha * safe_log_ratio - 1) + torch.exp(alpha * log_q),
            dim=-1,
        ) / (alpha ** 2)
    elif abs(alpha + beta) < eps:
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1, dim=-1) / (alpha ** 2)
    else:
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = -(term1 - term2 - term3) / (alpha * beta)

    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    return _masked_mean(safe_divergence, no_model_batch)


def bdkd(logits, teacher_logits, no_model_batch):
    def entropy(x):
        probs = F.softmax(x, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        return -torch.sum(probs * log_probs, dim=-1)

    entropy_student = entropy(logits)
    entropy_teacher = entropy(teacher_logits)
    weight_student = torch.where(entropy_student > entropy_teacher, 2.0, 1.0)
    weight_teacher = torch.where(entropy_teacher > entropy_student, 2.0, 1.0)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    mask_sum = torch.sum(mask.view(-1), dim=0)
    distil_loss1 = -torch.sum(x * mask.view(-1) * weight_teacher.view(-1), dim=0) / mask_sum

    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss2 = -torch.sum(x * mask.view(-1) * weight_student.view(-1), dim=0) / mask_sum

    return distil_loss1 + distil_loss2


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    return (1 - lam) * for_kl + lam * rev_kl


def get_ratio(teacher_logits, logits, mu=0.5):
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)
    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)
    mask = cum_sum > mu
    mask[:, :, 0] = False

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)
    denom = (s1 + s2).clamp(min=1e-8)
    return s1 / denom, s2 / denom


def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)

    if ratio is None:
        weights = mask.view(-1)
    else:
        weights = ratio.view(-1) * mask.view(-1)
    return torch.sum((teacher_x - x) * weights, dim=0) / torch.sum(mask.view(-1), dim=0)


def AKL(teacher_logits, logits, no_model_batch):
    inf_mask = torch.isinf(logits)
    mask = (no_model_batch["label"] != -100).int()
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    return get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(
        logits, teacher_logits, inf_mask, mask, l_ratio
    )


def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def wsd(logits, teacher_logits, no_model_batch, lam=0.5):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    return (1 - lam) * for_kl + lam * rev_kl


def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def csd(logits, teacher_logits, no_model_batch, mode="SS"):
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    if mode == "SS":
        loss = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits),
            dim=-1, keepdim=True)).detach() * student_probs.detach() * logits
    elif mode == "TS":
        loss1 = (logits - teacher_logits - torch.sum(teacher_probs * (logits - teacher_logits),
            dim=-1, keepdim=True)).detach() * student_probs.detach() * logits
        loss2 = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits),
            dim=-1, keepdim=True)).detach() * teacher_probs * logits
        loss = (loss1 + loss2) / 2
    else:
        raise ValueError(f"Unsupported CSD mode: {mode}")

    x = torch.sum(loss, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    inf_mask = torch.isinf(q_logits) | torch.isinf(p_logits)
    q_logits = torch.masked_fill(q_logits, inf_mask, 0)
    p_logits = torch.masked_fill(p_logits, inf_mask, 0)
    q_prob = F.softmax(q_logits, dim=-1).detach()
    p_prob = F.softmax(p_logits, dim=-1).detach()
    q_log_prob = F.log_softmax(q_logits, dim=-1)

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha).clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=-1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=-1)
    return loss, grad_loss


def alphanet(logits, teacher_logits, no_model_batch, alpha, beta):
    loss1 = ab_div(logits, teacher_logits, no_model_batch, alpha, 1 - alpha)
    loss2 = ab_div(logits, teacher_logits, no_model_batch, beta, 1 - beta)
    return loss1 if loss1 > loss2 else loss2


def amid(logits, teacher_logits, no_model_batch, args, **kwargs):
    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    logp = F.log_softmax(teacher_logits, dim=-1)
    logq = F.log_softmax(logits, dim=-1)

    alpha = args.amid_alpha
    lam = args.amid_lam
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)

    if lam <= 0.0:
        r = q
        logr = F.log_softmax(logits, dim=-1)
    elif lam >= 1.0:
        r = p
        logr = F.log_softmax(teacher_logits, dim=-1)
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
        else:
            t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)

    div_name = args.amid_div_name
    div_order = args.amid_div_order
    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"Unsupported AMID div_order: {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        return torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    if div_name == "ab":
        ab_alpha, ab_beta = 0.5, 0.5
        apb = ab_alpha + ab_beta
        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
        else:
            raise ValueError(f"Unsupported AMID div_order: {div_order}")
        divergence = -(term1 - term2 - term3) / (ab_alpha * ab_beta)
        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        return masked_sum / mask_sum if mask_sum > 0 else masked_sum

    raise ValueError(f"Unsupported AMID div_name: {div_name}")
