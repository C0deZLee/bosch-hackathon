from django.contrib import admin
from django.urls import reverse
from django.utils.safestring import mark_safe

from .models import Message

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
	# List display Settings
	list_display = ('id', 'time')
	search_fields = ('created',)
	ordering = ('created',)