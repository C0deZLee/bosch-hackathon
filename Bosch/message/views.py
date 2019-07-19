from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser

class MessageViewSet(viewsets.ViewSet):
	parser_classes = (FormParser, JSONParser, MultiPartParser)

	def message(self, request):
		"""
		Retrieve user wallet balance info
		"""
		# Get the cached balance info from database
		# TODO
		# serializer = FullWalletSerializer(request.user.investor.wallet)
		return Response({'msg':'test'})

