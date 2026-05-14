import os
import subprocess
from neo4j import GraphDatabase, Query, NotificationDisabledClassification
from neo4j.exceptions import Neo4jError
from typing import Literal, Tuple, List, Dict, Any
import time
import json
from src.logger_config import setup_logger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logger
logger = setup_logger(__name__)

class Neo4jConnector:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str = "neo4j",
        name: str = "neo4j-db",
        max_connection_pool_size: int = 100,
        debug: bool = False
    ):
        """A clean, robust Neo4j connector with query + notification logging."""
        self.name = name
        self.database = database
        self.debug = debug

        self.driver = GraphDatabase.driver(
            f"{host}:{port}",
            auth=(username, password),
            max_connection_pool_size=max_connection_pool_size,
            # notifications_disabled_classifications=list(NotificationDisabledClassification)
        )
        # self.driver.verify_connectivity()

    def run_query(
        self,
        cypher,
        timeout=None,
        convert_func: Literal['data', 'graph'] = 'data',
        **kwargs
    ):
        if self.debug:
            t0 = time.time()
            logger.info(f'Running Cypher:\n```\n{cypher}\n```')
        try:
            with self.driver.session(database=self.database) as session:
                query = Query(cypher, timeout=timeout)
                result = session.run(query, **kwargs)
                if convert_func == 'data':
                    result = result.data()
                elif convert_func == 'graph':
                    result = result.graph()
                else:
                    raise ValueError(f"Invalid convert_func: {convert_func}")

        except Exception as e:
            # Log lỗi để trace
            logger.error(f"ERROR when executing Cypher!: '{cypher}'")
            raise e
        if self.debug:
            logger.info("Query finished in %.2fs", time.time() - t0)
        return result

    def run_query_advance(
        self,
        cypher: Any,
        *,
        timeout: int = None,
        convert_func: Literal["list", "data", "graph"] = "list",
        **kwargs
    ) -> Dict:
        if self.debug:
            logger.info("==== RUN QUERY ====")
            logger.info(cypher)
            logger.info("Params: %s", params)
            t0 = time.time()

        try:
            with self.driver.session(database=self.database) as session:

                query = Query(cypher, timeout=timeout)
                result = session.run(query, **kwargs)

                # -------- FETCH RECORDS BEFORE CONSUME --------
                if convert_func == "list":
                    records = [dict(r) for r in result]
                elif convert_func == "data":
                    records = result.data()
                elif convert_func == "graph":
                    records = result.graph()
                else:
                    raise ValueError("Invalid convert value")

                # -------- THEN CONSUME FOR NOTIFICATIONS --------
                summary = result.consume()

                notifications = []
                has_error = False
                for note in summary.gql_status_objects:
                    notifications.append({
                        "severity": note.severity.name,
                        "gql_status": note.gql_status,
                        "description": note.status_description,
                        "position": {
                            "line": getattr(note.position, "line", None),
                            "column": getattr(note.position, "column", None),
                            "offset": getattr(note.position, "offset", None),
                        },
                        "classification": str(note.classification),
                        "raw_classification": note.raw_classification,
                    })
                    if note.severity.name in {"ERROR", "FATAL", "WARNING"}:
                        has_error = True
                if self.debug:
                    logger.info("Query OK in %.3fs", time.time() - t0)
                    if notifications:
                        logger.warning("Notifications: %s", notifications)

                return {
                    "success": not has_error,
                    "records": records,
                    "notifications": notifications
                }

        # except Neo4jError as e:
        #     logger.error("=== Neo4j ERROR ===")
        #     logger.error("Code: %s", e.code)
        #     logger.error("Message: %s", e.message)
        #     import traceback
        #     traceback.print_exc()
        #     return {
        #         "success": False,
        #         "records": [],
        #         "notifications": [{"severity": "EXCEPTION", "description": str(e)}]
        #     }
        except Exception as e:
            # Log lỗi để trace
            logger.error(f"ERROR when executing Cypher!: '{cypher}'")
            return {
                "success": False,
                "records": [],
                "notifications": [{"severity": "EXCEPTION", "description": str(e)}]
            }

    def get_num_entities(self):
        return self.run_query("MATCH (n) RETURN count(n) as num")[0]['num']

    def get_num_relations(self):
        return self.run_query("MATCH ()-[r]->() RETURN count(r) as num")[0]['num']

    def wait_for_db_online(self, db_name, timeout=10):
        """Wait until DB is online."""
        with self.driver.session(database="system") as session:
            start = time.time()
            while True:
                dbs = session.run("SHOW DATABASES").data()
                for db in dbs:
                    if db["name"] == db_name and db["currentStatus"].lower() == "online":
                        logger.info("Database '%s' is online!", db_name)
                        return
                if time.time() - start > timeout:
                    raise TimeoutError(f"Database '{db_name}' not online after {timeout} seconds")
                time.sleep(0.5)

    def create_or_reset_db(self, db_name: str, overwrite: bool):
        """Create DB if not exist. If overwrite=True, drop then recreate."""
        with self.driver.session(database="system") as session:
            dbs = session.run("SHOW DATABASES").data()
            exists = any(db["name"] == db_name for db in dbs)

            if exists and overwrite:
                logger.info("Dropping existing database: '%s'", db_name)
                session.run(f"DROP DATABASE {db_name} IF EXISTS")
                session.run(f"CREATE DATABASE {db_name}")
            elif not exists:
                logger.info("Creating new database: '%s'", db_name)
                session.run(f"CREATE DATABASE {db_name}")
            else:
                logger.info("Database '%s' already exists — skip (overwrite=False)", db_name)

    @staticmethod
    def apply_schema_constraints(session, schema):
        """
        Create UNIQUE constraint on (label, eid)
        """
        unique_labels = {e["label"] for e in schema["entities"]}

        for label in unique_labels:
            cypher = f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:`{label}`)
                REQUIRE n.eid IS UNIQUE
                """
            session.run(cypher)
            logger.info("Constraint ensured for label: %s", label)

    @staticmethod
    def import_entities(session, entities, batch_size=5000):
        """
        Import nodes grouped by label
        """
        from collections import defaultdict

        grouped = defaultdict(list)
        for ent in entities:
            grouped[ent["label"]].append({
                "eid": ent["eid"],
                "name": ent.get("name"),
                "aliases": ent.get("aliases", []),
                "description": ent.get("description"),
                "provenance": ent.get("provenance", []),
                "properties": ent.get("properties", {})
            })

        query_tpl = """
        UNWIND $batch AS row
        MERGE (n:`{label}` {{eid: row.eid}})
        SET
            n.name        = row.name,
            n.aliases     = row.aliases,
            n.description= row.description,
            n.provenance = row.provenance
        SET n += row.properties
        """

        for label, rows in grouped.items():
            cypher = query_tpl.format(label=label)

            for i in range(0, len(rows), batch_size):
                session.run(cypher, batch=rows[i:i+batch_size])

            logger.info("Imported %d nodes [%s]", len(rows), label)

    def _import_relations_single_label(
        self,
        db_name,
        label,
        relations,
        source_label,
        target_label,
        batch_size=5000
    ):
        """
        Import relations of ONE type (label) using CREATE
        Single writer → minimal locking
        """

        cypher = f"""
        UNWIND $batch AS row
        MATCH (s:`{source_label}` {{eid: row.subj}})
        MATCH (o:`{target_label}` {{eid: row.obj}})
        CREATE (s)-[r:`{label}`]->(o)
        SET
            r += row.properties,
            r.provenance = row.provenance
        """

        with self.driver.session(database=db_name) as session:
            tx = session.begin_transaction()

            for i in range(0, len(relations), batch_size):
                batch = relations[i:i+batch_size]
                tx.run(cypher, batch=batch)

                # Commit mỗi batch để tránh transaction quá lớn
                tx.commit()
                tx = session.begin_transaction()

            tx.commit()

        logger.info("Imported %d relationships [%s]", len(relations), label)

    def import_relations(
        self,
        relations,
        db_name,
        label_mapping,
        batch_size=5000,
        max_workers=4
    ):
        """
        label_mapping:
        {
          "REL_TYPE": {
              "source": "EntityA",
              "target": "EntityB"
          }
        }
        """
        grouped = defaultdict(list)

        for rel in relations:
            grouped[rel["label"]].append({
                "subj": rel["subj_id"],
                "obj": rel["obj_id"],
                "properties": rel.get("properties", {}),
                "provenance": rel.get("provenance", [])
            })

        tasks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for label, rows in grouped.items():
                cfg = label_mapping[label]

                future = executor.submit(
                    self._import_relations_single_label,
                    db_name,
                    label,
                    rows,
                    cfg["source"],
                    cfg["target"],
                    batch_size
                )
                tasks.append(future)

            for f in as_completed(tasks):
                f.result()

    @staticmethod
    def build_label_mapping(schema):
        return {
            r["label"]: {
                "source": r["subj_label"],
                "target": r["obj_label"]
            }
            for r in schema.get("relations", [])
        }

    def import_json_dataset(
        self,
        filepath,
        db_name=None,
        overwrite=False,
        entity_batch=5000,
        rel_batch=5000
    ):
        db = db_name or self.database

        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)

        schema = data["schema"]
        entities = data["entities"]
        relations = data["relations"]

        label_mapping = self.build_label_mapping(schema)

        # Create/reset database
        self.create_or_reset_db(db, overwrite)
        self.wait_for_db_online(db)

        with self.driver.session(database=db) as session:
            self.apply_schema_constraints(session, schema)
            self.import_entities(session, entities, batch_size=entity_batch)

        self.import_relations(
            relations,
            db_name=db,
            label_mapping=label_mapping,
            batch_size=rel_batch
        )

        logger.info("IMPORT DONE for DB: %s", self.database)
