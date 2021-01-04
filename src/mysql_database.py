import mysql.connector

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="123456789",
#   database="LicensePlate"
# )


# mycursor = mydb.cursor()

# mycursor.execute("SELECT * FROM LicensePlate")
# for x in mycursor:
#   print(x[1])

# def insert_db(string_lp, lptime):
#     querry   = "INSERT INTO c+ (lpname, time) VLUES (%s, %s)  "
#     val = (string_lp, lptime)
#     mydb.excute(querry, val)
#     mydb.commit()
#     print(mycursor.rowcount, "record inserted.")

# def querry_db(string_lp):
#     mydb.excute("SELECT * FROM License_PlateDB")
#     myresult_querry = mydb.fetchall()
#     datetime_querry = ''
#     for value in myresult_querry:
#         if (value[2] == string_lp):
#             datetime_querry = value[1]
#             break
#     return datetime_querry

# def insert_Plate_IN(string_lp, lptime):
#     querry = "INSERT INTO LicensePlate (LicensePlate_Name, TIME_IN, "
    
    
class MySQL_Database():
    def __init__(self):
        self.mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="123456789",
                    database="LicensePlate"
                    )
        self.mycursor = self.mydb.cursor(buffered=True)
    
    def insert_Plate_IN(self, string_lp, lptime):
        querry = "INSERT INTO LicensePlate (LicensePlate_Name, TIME_IN, status) VALUES (%s, %s,%s)"
        if len(string_lp) != 0:
            val = (string_lp, lptime, "IN")
            self.mycursor.execute(querry,val)
            self.mydb.commit()
        
    def querry_Plate_OUT(self, string_lp, lptime):
        querry = "SELECT * FROM LicensePlate WHERE LicensePlate_Name = %s AND status = %s"
        if len(string_lp) != 0:
            val = (string_lp, "IN")
            print(string_lp)
            self.mycursor.execute(querry,val)
            myresult = self.mycursor.fetchone()
            print(myresult)
            if myresult != None:
                print(myresult)
                querry = "UPDATE LicensePlate SET status = %s, TIME_OUT = %s WHERE ID = %s"
                val = ("DONE", lptime, myresult[0])
                self.mycursor.execute(querry,val)
                self.mydb.commit()
                return myresult[2]

        return None
            
            
    def select_all(self):
        querry = "SELECT * FROM LicensePlate"
        self.mycursor.execute(querry)
        for line in self.mycursor:
            print(line)
    
if __name__ == "__main__":
    myDB = MySQL_Database()
    myDB.querry_Plate_OUT("56C-1asd3212", "11/11/2020 12:00:00")
    #myDB.select_all()
    # myDB.querry_plate_out("56C-132112", "11/11/2020 12:00:00")