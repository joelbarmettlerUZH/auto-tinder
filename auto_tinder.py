import requests
import datetime
# from geopy.geocoders import Nominatim
from time import sleep
from random import random
import person_detector
from time import time
import random
import logging
import datetime
import urllib.request
import os



TINDER_URL = "https://api.gotinder.com"
# geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "./images/unclassified/profiles.txt"
IMAGE_DOWNLOAD_FOLDER = "/home/michael/Developement/dating-robot/auto-tinder-images/"

# api interface

class tinderAPI():

    def __init__(self, token):
        self._token = token
        self.name = "tinderAPI"

    def like(self, user_id):
        print("like user {0}".format(user_id))
        data = requests.get(TINDER_URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        print("pass user {0}".format(user_id))        
        requests.get(TINDER_URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True
            
    def map_user_to_person(self, user, api):
        id = user["_id"]
        name =  user["name"]
        photo_urls = list(map(lambda photo: photo["url"], user["photos"]))
        return Person(api, id, name, photo_urls)  

    def nearby_persons(self):
        data = requests.get(TINDER_URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        users = data["data"]["results"]
        # persons = list(map(lambda user: self.map_user_to_person(user["user"], self), users))
        persons = []
        for user in users:
                person = self.map_user_to_person(user["user"], self)
                persons.append(person)
                
        return persons


    def profile(self):
        data = requests.get(TINDER_URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(TINDER_URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

   

class Person(object):

    def __init__(self, api, id, name, photo_urls):
        self._api = api

        self.id = id
        self.name = name
        self.images = photo_urls

    # self.bio = data.get("bio", "")
    # self.distance = data.get("distance_mi", 0) / 1.60934

    # self.birth_date = datetime.datetime.strptime(data["birth_date"], '%Y-%m-%dT%H:%M:%S.%fZ') if data.get(
    #     "birth_date", False) else None
    # self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]

    

    # self.jobs = list(
    #     map(lambda job: {"title": job.get("title", {}).get("name"), "company": job.get("company", {}).get("name")}, data.get("jobs", [])))
    # self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

    # if data.get("pos", False):
    #     self.location = geolocator.reverse(f'{data["pos"]["lat"]}, {data["pos"]["lon"]}')

# def __repr__(self):
#     return f"{self.id}  -  {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"
    
    def like(self):        
        return self._api.like(self.id)

    def dislike(self):
        return self._api.dislike(self.id)

    def download_images(self):
        # with open(PROF_FILE, "r") as f:
        #     lines = f.readlines()
        #     if self.id in lines:
        #         return
        # with open(PROF_FILE, "a") as f:
        #     f.write(self.id+"\r\n")
        # index = -1
        # for image_url in self.images:
        #     index += 1
        #     req = requests.get(image_url, stream=True)
        #     if req.status_code == 200:
        #         file_full_path = f"{folder}/{self.id}_{self.name}_{index}.jpeg"
        #         with open( file_full_path, "wb") as f:
        #             f.write(req.content)
        #     sleep(random()*sleep_max_for)
        url = self.images[0]
        file_full_path = f"{IMAGE_DOWNLOAD_FOLDER}/{self.id}_{self.name}_0.jpeg"
        is_existed = os.path.isfile(file_full_path)
        if(is_existed):
            file_full_path = f"{IMAGE_DOWNLOAD_FOLDER}/{self.id}_{self.name}_1.jpeg"
        urllib.request.urlretrieve(url,file_full_path)
        sleep(random.uniform(0, 1))


class Profile(Person):

    def __init__(self, data, api):

        super().__init__(data["user"], api)

        # self.email = data["account"].get("email")
        # self.phone_number = data["account"].get("account_phone_number")

        # self.age_min = data["user"]["age_filter_min"]
        # self.age_max = data["user"]["age_filter_max"]

        self.max_distance = data["user"]["distance_filter"]
        # self.gender_filter = ["Male", "Female"][data["user"]["gender_filter"]]


# add desicion making algo
def predict_likeliness(person):
    return random.uniform(0, 1)


if __name__ == "__main__":
    token = "145773e4-4e44-4a59-8d94-ac577f8d0dcf"
    api = tinderAPI(token)
    batch_count = 0 
    person_count = 0

    # sleep(random.uniform(0, 1) * 3)

    datestr = datetime.date.today().strftime("%Y%m%d")
    logfile = f"/home/michael/Developement/dating-robot/auto-tinder-log/auto_tinder_{datestr}.log"

    logging.basicConfig(filename=logfile, filemode = 'a', level=logging.DEBUG, format = '%(asctime)s - %(levelname)s: %(message)s',\
                     datefmt = '%m/%d/%Y %I:%M:%S %p' )

    # logging.debug('This message should go to the log file')
    logging.info('start')
    # logging.warning('And this, too')
    # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

    while batch_count < 5:        
            
        persons = api.nearby_persons()
        print(f"--------------------------------Batch: {batch_count}. Get new profiles. batch_counts: {len(persons)} -------------------------")
        logging.info(f"Batch: {batch_count}. Get new profiles. batch_counts: {len(persons)} ")

        for person in persons:                
            score = predict_likeliness(person)                    

            print("-------------------------")
            print("ID: ", person.id)
            print("Name: ", person.name)                
            print("Images: ", person.images[0])
            print("Score:", score)

            sleep(random.uniform(0, 1) * 8)
            person.download_images()
            try:
                if score > 0.35:                        
                    logging.info(f"Liked person: {person.id}")
                    res = person.like()
                    # print("LIKE")
                    # print("Response: ", res)
                else:
                    logging.info(f"Passed person: {person.id}")
                    res = person.dislike()
                    # print("DISLIKE")
                    # print("Response: ", res)
            except Exception:
                logging.error(f'Can not submit judgement to {person.api.name}')    

            person_count = person_count + 1     

        batch_count = batch_count + 1

    print(f'success proccessed profiles: {person_count}')    
    logging.info(f'success proccessed profiles: {person_count}')    

          