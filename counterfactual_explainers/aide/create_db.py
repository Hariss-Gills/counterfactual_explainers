import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """create a database connection to a database that resides
    in the memory
    """
    # db_file = 'paper_one/explanation_store/ex_store.db'
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn


def create_db(db_file):
    conn = create_connection(db_file)
    create_table_attribute_types(conn)
    create_table_operands(conn)
    create_table_attributes_operands_types(conn)
    create_table_attributes(conn)
    create_table_explanations(conn)
    create_table_deltas(conn)
    create_table_rules(conn)
    create_table_records(conn)
    create_table_importances(conn)
    # new TABLE
    create_table_class_probabilities(conn)
    create_table_attributes_deltas(conn)
    create_table_records_attributes(conn)
    create_table_importances_attributes(conn)
    create_table_rules_attributes(conn)
    create_table_differences_attributes(conn)
    create_table_invariants_attributes(conn)
    # new table
    create_table_class_probabilities_attributes(conn)


def create_table_attribute_types(conn):
    sql = """CREATE TABLE IF NOT EXISTS `attribute_types` (
	`attribute_type_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
	`attribute_name`	TEXT
    );    """
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_operands(conn):
    sql = """CREATE TABLE IF NOT EXISTS `operands` (
    	`operand_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
    	`operand_value`	TEXT
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_attributes_operands_types(conn):
    sql = """CREATE TABLE IF NOT EXISTS `attributes_operands_types` (
    	`attributes_operands_types_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
    	`operand_id`	INTEGER,
    	`attribute_type_id`	INTEGER,
    	FOREIGN KEY(`attribute_type_id`) REFERENCES `attribute_types`(`attribute_type_id`)
        FOREIGN KEY(`operand_id`) REFERENCES `operands`(`operand_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `attributes` (
    	`attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
    	`attributes_operands_types_id` INTEGER,
    	`attribute_value` TEXT,
    	FOREIGN KEY(`attributes_operands_types_id`) REFERENCES `attributes_operands_types`(`attributes_operands_types_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_explanations(conn):
    sql = """CREATE TABLE IF NOT EXISTS `explanations` (
    	`explanation_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        `description`	TEXT,
        `record_number` INTEGER
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_deltas(conn):
    sql = """CREATE TABLE IF NOT EXISTS `deltas` (
        	`delta_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`explanation_id`	INTEGER,
            FOREIGN KEY(`explanation_id`) REFERENCES `explanations`(`explanation_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_rules(conn):
    sql = """CREATE TABLE IF NOT EXISTS `rules` (
       `rule_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
       `explanation_id`	INTEGER,
       FOREIGN KEY(`explanation_id`) REFERENCES `explanations`(`explanation_id`)
       );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_records(conn):
    sql = """CREATE TABLE IF NOT EXISTS `records` (
        	`record_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`explanation_id`	INTEGER,
            FOREIGN KEY(`explanation_id`) REFERENCES `explanations`(`explanation_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_importances(conn):
    sql = """CREATE TABLE IF NOT EXISTS `importances` (
        	`importance_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`explanation_id`	INTEGER,
            FOREIGN KEY(`explanation_id`) REFERENCES `explanations`(`explanation_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_class_probabilities(conn):
    sql = """CREATE TABLE IF NOT EXISTS `class_probabilities` (
        	`class_probability_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`explanation_id`	INTEGER,
            FOREIGN KEY(`explanation_id`) REFERENCES `explanations`(`explanation_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_attributes_deltas(conn):
    sql = """CREATE TABLE IF NOT EXISTS `attributes_deltas` (
        	`attribute_delta_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`delta_id`	INTEGER,
        	`attribute_id`	INTEGER,
            FOREIGN KEY(`attribute_id`) REFERENCES `attributes`,
        	FOREIGN KEY(`delta_id`) REFERENCES `deltas`
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_differences_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `differences_attributes` (
        	`difference_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`delta_id`	INTEGER,
        	`attribute_id`	INTEGER,
            FOREIGN KEY(`attribute_id`) REFERENCES `attributes`,
        	FOREIGN KEY(`delta_id`) REFERENCES `deltas`
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_records_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `records_attributes` (
        	`record_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`record_id`	INTEGER,
        	`attribute_id`	INTEGER,
            FOREIGN KEY(`attribute_id`) REFERENCES `attributes`,
        	FOREIGN KEY(`record_id`) REFERENCES `records`(`record_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_invariants_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `invariants_attributes` (
        	`invariant_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`record_id`	INTEGER,
        	`attribute_id`	INTEGER,
            FOREIGN KEY(`attribute_id`) REFERENCES `attributes`,
        	FOREIGN KEY(`record_id`) REFERENCES `records`(`record_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_importances_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `importances_attributes` (
        	`importance_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`attribute_id`	INTEGER,
        	`importance_id`	INTEGER,
        	FOREIGN KEY(`attribute_id`) REFERENCES `attributes`(`attribute_id`),
        	FOREIGN KEY(`importance_id`) REFERENCES `attributes`(`attribute_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_rules_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `rules_attributes` (
        	`rule_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`rule_id`	INTEGER,
        	`attribute_id`	INTEGER,
        	FOREIGN KEY(`rule_id`) REFERENCES `rules`(`rule_id`),
        	FOREIGN KEY(`attribute_id`) REFERENCES `attributes`(`attribute_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)


def create_table_class_probabilities_attributes(conn):
    sql = """CREATE TABLE IF NOT EXISTS `class_probabilities_attributes` (
        	`class_probability_attribute_id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        	`class_probability_id`	INTEGER,
        	`attribute_id`	INTEGER,
        	FOREIGN KEY(`class_probability_id`) REFERENCES `class_probabilities`(`class_probability_id`),
        	FOREIGN KEY(`attribute_id`) REFERENCES `attributes`(`attribute_id`)
        );"""
    try:
        c = conn.cursor()
        c.execute(sql)
    except Error as e:
        print(e)
