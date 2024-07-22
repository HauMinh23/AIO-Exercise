# Homework 2
from collections import Counter
from abc import ABC, abstractmethod


class Person(ABC):
    def __init__(self, name, yob):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name=name, yob=yob)
        self._grade = grade

    def describe(self):
        print(
            f"Student - Name : {self._name} - YoB: {self._yob} - Grade: {self._grade}")


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name=name, yob=yob)
        self._specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name : {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name=name, yob=yob)
        self._subject = subject

    def describe(self):
        print(
            f"Teacher - Name : {self._name} - YoB: {self._yob} - Subject: {self._subject}")


Student1 = Student("Thuan", 2011, "10")
Student1.describe()

doctor1 = Doctor("Thien", 1998, "Hi")
doctor1.describe()

doctor2 = Doctor("Thu", 1987, "Hello")
doctor2.describe()

teacher1 = Teacher("Thien", 1998, "Math")
teacher1.describe()


class Ward:
    def __init__(self, name):
        self.__name = name
        self.__list_people = list()

    def add_person(self, person: Person):
        self.__list_people.append(person)

    def describe(self):
        print(f"Name: {self.__name}")
        for p in self.__list_people:
            p.describe()

    def count_doctor(self):
        counter = 0
        for p in self.__list_people:
            if isinstance(p, Doctor):
                counter += 1
        return counter

    def sort_yob(self):
        return self.__list_people.sort(key=lambda x: x.get_yob(), reverse=True)

    def compute_average(self):
        total = 0
        for p in self.__list_people:
            total += p.get_yob()
        return total/len(self.__list_people)

    ward1 = Ward("Ward1")   # type: ignore

    ward1.add_person(Student1)


ward1 = Ward("Ward1")

ward1.add_person(Student1)
ward1.describe()

ward1.add_person(doctor1)
ward1.describe()

ward1.add_person(teacher1)
ward1.describe()

ward1 = Ward("Ward1")
ward1.add_person(doctor1)
ward1.add_person(doctor2)
ward1.describe()

ward1.count_doctor()

ward1.sort_yob()
ward1.describe()

ward1.compute_average()
