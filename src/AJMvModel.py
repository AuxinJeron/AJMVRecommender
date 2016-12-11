class Movie:
    def __init__(self, id, name, categories):
        self.id = id
        self.name = name
        self.categories = categories

    def __str__(self):
        return "[id: {} name: {} categories: {}]".format(self.id, self.name, self.categories)

    def __repr__(self):
        return "[id: {} name: {} categories: {}]".format(self.id, self.name, self.categories)

class User:
    def __init__(self, id, gender, age, occupation):
        self.id = id
        self.gender = gender
        self.age = age
        self.occupation = occupation

    def __str__(self):
        return "[id: {} gender: {} age: {} occupation: {}]".format(self.id, self.gender, self.age, self.occupation)

    def __repr__(self):
        return "[id: {} gender: {} age: {} occupation: {}]".format(self.id, self.gender, self.age, self.occupation)
