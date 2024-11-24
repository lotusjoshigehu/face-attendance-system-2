import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceaccount.json")
firebase_admin.initialize_app(cred,{'databaseURL': "https://faceattendencerealtime-f66ae-default-rtdb.firebaseio.com//"})
ref=db.reference("Students")

data={
    "12346":
        {
            "name": "Karan Chauhan",
            "major":"Cs",
            "starting year":2022,
            "total attendance":365,
            "standing":"Good",
            "year":3,
            "last_attendance_time":'2024-07-12 00:54:34'
        },
    "12345":
        {
            "name": "Vicky Kaushal",
            "major":"Acting",
            "starting year":2022,
            "total attendance":200,
            "standing":"bad",
            "year":2,
            "last_attendance_time":'2024-07-12 00:54:34'
        },
    "12344":
        {
            "name": "Kamal Joshi",
            "major":"Cs",
            "starting year":2022,
            "total attendance":290,
            "standing":"Good",
            "year":3,
            "last_attendance_time":'2024-07-12 00:54:34'
        }

}

for key,value in data.items():
    ref.child(key).set(value)