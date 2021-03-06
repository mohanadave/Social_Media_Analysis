from project.com.dao import conn_db
import json
import copy
conn = conn_db()

class loginDAO:
    def login_user(self, user):
        cursor = conn.cursor()
        query="SELECT * FROM users WHERE username='{}'".format(user.username)
        cursor.execute(query)
        data = cursor.fetchall()
        conn.commit()
        cursor.close()
        return data

class registerDAO:
    def do_necessary_changes(self, user):
        cursor=conn.cursor()
        file_ptr = open('data/annoted_tweets.json')
        users = json.load(file_ptr)
        file_ptr.close()
        temp = copy.deepcopy(users['admin'])
        temp['annoted_tweets'] = []
        temp['reported_tweets'] = []
        temp['last_updated'] = None
        temp['start_end'] = [int(user.start), int(user.end)]
        users[user.username] = temp
        file_ptr = open('data/annoted_tweets.json','w')
        data_dumped=json.dumps(users,indent=4)
        file_ptr.write(data_dumped)
        file_ptr.close()
        print('Json file updated')
        query="alter TABLE annoted_tweets add {} TEXT;".format(user.username)
        cursor.execute(query)
        conn.commit()
        cursor.close()
        print('database Updated')




    def register_user(self, user):
        cursor = conn.cursor()
        query="INSERT INTO users (name,username,password) VALUES ('{}','{}','{}') ".format(user.name,user.username,user.password)
        cursor.execute(query)
        conn.commit()
        cursor.close()
        return True

    def check_exists(self,user):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username='{}'".format(user.username))
        data = cursor.fetchall()
        conn.commit()
        cursor.close()

        return data