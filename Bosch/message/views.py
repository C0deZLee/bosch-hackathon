from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser, FileUploadParser

from .models import Message
from .serializers import MessageSerializer

class MessageViewSet(viewsets.ViewSet):
	parser_classes = (JSONParser,)

	def predict(self, request):
		msg = Message(
			x = request.data.x,
			y = request.data.y,
			z = request.data.z,
			time = request.data.time,
			label = '')
		msg.save()
		
		# save

		return Response({ 'data': 1, 'code': 200, 'id': ''})

	def reinforce(self, request):
		id = request.id
		label = request.label

		msg = Message.objects.get(id=id)
		
		# call reinforce function

		return Response(status=status.HTTP_200_OK)

	def all(self, request):
		msgs = Message.objects.all()
		serializer = FullCompanyUserSerializer(msgs, many=True)
		return Response(serializer.data, status=status.HTTP_200_OK)