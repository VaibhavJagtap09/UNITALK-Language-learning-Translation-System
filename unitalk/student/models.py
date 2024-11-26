from django.db import models

# Create your models here.
class Student(models.Model):
    student_name = models.CharField(max_length=200)
    username = models.CharField(max_length=200)
    mobile_no = models.BigIntegerField()
    email_id = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    confirm_password = models.CharField(max_length=200)

