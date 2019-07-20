from rest_framework import serializers

from .models import Message

class MessageSerializer(serializers.ModelSerializer):
	class Meta:
		model = Message
		fields = ('id', 'x', 'y', 'z', 'time', 'created',)
		read_only_fields = ('created',)

