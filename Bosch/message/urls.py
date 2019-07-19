from django.urls import path

from .views import MessageViewSet

message = MessageViewSet.as_view({
	'post': 'message',
})

urlpatterns = [
	path('message/', message, name='message'),
]
