import sqlite3

def createDB():
    conn = sqlite3.connect('digits.sqlite')
    c = conn.cursor()
    c.execute('CREATE TABLE digits (filename text, output int)')
    conn.commit()
    conn.close()

def fetchAll():
    conn = sqlite3.connect('digits.sqlite')
    c = conn.cursor()
    c.execute('SELECT * FROM digits')
    print(c.fetchall())
    conn.commit()
    conn.close()

def deleteAllRows():
    conn = sqlite3.connect('digits.sqlite')
    c = conn.cursor()
    c.execute('DELETE FROM digits')
    conn.commit()
    conn.close()

def insertData(filename,out):
    conn = sqlite3.connect('digits.sqlite')
    c = conn.cursor()
    c.execute('INSERT INTO digits (filename,output) VALUES (?,?)',(filename,out))
    c.execute('SELECT * FROM digits')
    if(len(c.fetchall()) % 5 == 0):
        import classifier
        cl = classifier.classifier()
        cl.retrainClassifier()
    conn.commit()
    conn.close()

if __name__ == "__main__":
    #createDB()
    fetchAll()
    #deleteAllRows()