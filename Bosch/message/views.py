from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser, FileUploadParser

class MessageViewSet(viewsets.ViewSet):
	parser_classes = (MultiPartParser,)

	def message(self, request):
		"""
		Retrieve user wallet balance info
		"""
		# Get the cached balance info from database
		# TODO
		# serializer = FullWalletSerializer(request.user.investor.wallet)
		if request.data:
			print(data)
		
		return Response({ 'data':'1', 'code':'200', 'message':"xxxxx", 'uploadeData': request.data})