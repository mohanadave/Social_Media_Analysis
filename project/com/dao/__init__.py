# Describe connection here
import sqlite3

def conn_db():
    return sqlite3.connect('socialmedia.db', check_same_thread=False)