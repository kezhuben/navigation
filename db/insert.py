#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: yangcheng
# license: © 2011-2017 The Authors. All Rights Reserved.
# contact: yangcheng@zuzuche.com
# time: 2017/11/10 17:10
# desc: insert data into database.
# ======================================================

import psycopg2


def single_insert(db_conf, table_schema, single_data, inner_use=False):
    """insert batch data into database table.

        Args:
            db_conf: `dict`, {database:, user:, password:, host:, port:}.
            table_schema: `dict`, {table_name:, columns_name:[...]}.
            single_data: `list`, [...].
            inner_use: `bool`, if inner use: True, else: False.

        Returns:
            inner_use = True => query sql.
            inner_use = False => Nothing.
    """
    if inner_use:
        db_tabel_name = table_schema['table_name']
        db_tabel_cols = table_schema['columns_name']
        insert_sql_front = """INSERT INTO {} {}""".format(db_tabel_name, tuple(db_tabel_cols)).replace("'", '"')
        insert_sql_back = """ VALUES {}""".format(tuple(single_data))
        insert_sql = insert_sql_front + insert_sql_back
        return insert_sql

    else:
        database = db_conf['database']
        user = db_conf['user']
        password = db_conf['password']
        host = db_conf['host']
        port = db_conf['port']

        db_tabel_name = table_schema['table_name']
        db_tabel_cols = table_schema['columns_name']

        db = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        cursor = db.cursor()

        insert_sql_front = """INSERT INTO {} {}""".format(db_tabel_name, tuple(db_tabel_cols)).replace("'", '"')
        insert_sql_back = """ VALUES {}""".format(tuple(single_data))
        insert_sql = insert_sql_front + insert_sql_back
        cursor.execute(insert_sql)
        db.commit()

        cursor.close()
        db.close()


def batch_insert(db_conf, table_schema, batch_data):
    """insert batch data into database table.

    Args:
        db_conf: `dict`, {database:, user:, password:, host:, port:}.
        table_schema: `dict`, {table_name:, columns_name:[...]}.
        batch_data: `list`, [[],[], ...].

    Returns:
        Nothing.
    """
    database = db_conf['database']
    user = db_conf['user']
    password = db_conf['password']
    host = db_conf['host']
    port = db_conf['port']

    db = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    cursor = db.cursor()

    for i in range(len(batch_data)):
        data = batch_data[i]
        insert_sql = single_insert(db_conf, table_schema, data, inner_use=True)
        cursor.execute(insert_sql)

    db.commit()
    cursor.close()
    db.close()


# if __name__ == '__main__':
#     DB_CONF = {
#         'database': "bigData",
#         'user': "postgres",
#         'password': "zzc709394",
#         'host': "121.46.20.195",
#         'port': "5432"
#     }
#     pg_table_schema = {
#         'table_name': "crossroad_images",
#         'columns_name': ["path", "label", "label_id", "road_data"]
#     }
#     batch_insert(DB_CONF, pg_table_schema, [['/data/bigData/121314235.png', 'right', '21', '[{test}]']])
