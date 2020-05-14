from django.db import models

# Create your models here.
class Document(models.Model):
	author = models.CharField(max_length=100)
	content = models.TextField()
	sentiment = models.CharField(max_length = 100)
	def __str__(self):
		return self.author
