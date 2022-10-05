import mysql.connector

from .CONFIG import Config


class DB(Config):
    def connect_db(self):
        try:
            db = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,
                database=self.database
            )
            return db
        except mysql.connector.Error as error:
            return "Something went wrong: {}".format(error)

