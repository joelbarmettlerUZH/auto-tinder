import json



class tinderAPI():
    def map_user_to_person(self, user, api):
        id = user["_id"]
        name =  user["name"]
        photo_urls = list(map(lambda photo: photo["url"], user["photos"]))
        return Person(api, id, name, photo_urls)    

    def nearby_persons(self):
        with open('dev_docs/tinder-recs-core-api-response.json') as f:
            json_data = json.load(f)
            users = json_data["data"]["results"]
            # return list(map(lambda user: self.map_user_to_person(user["user"], self), users))
            # # for
            persons = []
            for user in users:
                person = self.map_user_to_person(user["user"], self)
                persons.append(person)
            return persons

class Person(object):

    def __init__(self, api, id, name, photo_urls):
        self._api = api

        self.id = id
        self.name = name
        self.images = photo_urls

    
    def like(self):        
        return self._api.like(self.id)

    def dislike(self):
        return self._api.dislike(self.id)

    def download_images(self, folder="./images/raw", sleep_max_for=0):
        with open(PROF_FILE, "r") as f:
            lines = f.readlines()
            if self.id in lines:
                return
        with open(PROF_FILE, "a") as f:
            f.write(self.id+"\r\n")
        index = -1
        for image_url in self.images:
            index += 1
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                file_full_path = f"{folder}/{self.id}_{self.name}_{index}.jpeg"
                with open( file_full_path, "wb") as f:
                    f.write(req.content)
            sleep(random()*sleep_max_for)


api = tinderAPI()
persons = api.nearby_persons()

Print('success')
