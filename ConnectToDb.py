#import library
import psycopg2
import csv
import pandas as pd

def conn():
    try:
        #connect to data base
        conn = psycopg2.connect(host="localhost",database="IoT_Datawarehouse", user="postgres", password="admin")
        print('Connecting to the PostgreSQL database...')
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def findAllFact():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT s.id_subject ,sq.id_sequence,x_coordinate,y_coordinate,z_coordinate,sn.id_sensor,f.id_Activity ,instance_date FROM dim_subjects s INNER JOIN fact_table f ON s.id_subject = f.id_subject INNER JOIN dim_sequence sq ON f.id_sequence = sq.id_sequence INNER JOIN dim_coordinate c ON " \
    "c.id_coordinate = f.id_coordinate INNER JOIN dim_sensors sn ON sn.id_sensor = f.id_sensor INNER JOIN dim_activities a ON a.id_Activities= f.id_Activity"
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows

def findAllSensor():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT * FROM  dim_sensors "
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows
def findAllSequence():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT * FROM  dim_sequence "
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows

def findAllSubject():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT * FROM  dim_subjects "
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows

def findAllCoordinate():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT * FROM  dim_coordinate "
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows
def findAllActivities():
    # create a cursor
    cur = conn().cursor()
    # execute a statement
    sql = "SELECT * FROM  dim_activities "
    cur.execute(sql)
    rows = cur.fetchall()
    print("The number of parts: ", cur.rowcount)
    """for row in rows:
        print(row)"""
    cur.close()
    conn().close()
    return rows


def importTOcsv(a):
    listOFdata=a
    try:
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in listOFdata:
                writer.writerows([i])
        print("success")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
def importDataSegemntation():
    df_data = pd.read_csv("./output.csv",header=None)
    return df_data
def main():
    rows = findAllFact()
    importTOcsv(rows)

main()
"""importDataSegemntation()"""


