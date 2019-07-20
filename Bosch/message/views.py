from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser, FileUploadParser

from .models import Message
from .serializers import MessageSerializer
# from .web_itfc import predict, save_data, retrain

class MessageViewSet(viewsets.ViewSet):
	parser_classes = (JSONParser,)

	def predict(self, request):
		print(request.data)
		x = request.data["x"]
		y = request.data["y"]
		z = request.data["z"]
		time = request.data["time"]

		# save
		# label = predict(x, y, z, time)
		msg = Message(x = x, y = y, z = z, time = time, label = 'label')

		msg.save()

		return Response({ 'data': 'label', 'code': 200, 'id': msg.id})

	def reinforce(self, request):
		id = request.data["id"]
		label = request.data["label"]
		
		msg = Message.objects.get(id=id)
		msg.label = label
		msg.save()

		serializer = MessageSerializer(msg)

		print(msg.x, msg.y, msg.z, msg.time, label)
		# call reinforce function
		# save_data(msg["x"], msg["y"], msg["z"], msg["time"], label)
		return Response(serializer.data, status=status.HTTP_200_OK)

	def all(self, request):
		msgs = Message.objects.all()
		serializer = MessageSerializer(msgs, many=True)
		return Response(serializer.data, status=status.HTTP_200_OK)