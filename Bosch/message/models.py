from django.db import models

class Message(models.Model):
	"""
	This model is only for cache, record the request message for debug
	"""

	# Balance Info
	x = models.TextField()
	y = models.TextField()
	z = models.TextField()
	time = models.TextField()
	label = models.TextField()

	# TimeStamp
	created = models.DateTimeField(auto_now_add=True)

	class Meta:
		verbose_name_plural = 'Message'

	def __str__(self):
		return self.created.isoformat()