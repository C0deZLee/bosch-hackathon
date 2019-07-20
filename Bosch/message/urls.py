from django.urls import path

from .views import MessageViewSet

predict = MessageViewSet.as_view({
	'post': 'message',
})

reinforce = MessageViewSet.as_view({
	'post': 'reinforce',
})

all = MessageViewSet.as_view({
	'get': 'all',
})

urlpatterns = [
	path('predict/', predict, name='predict'),
	path('reinforce/', reinforce, name='reinforce'),
	path('all/', all, name='all'),
]
