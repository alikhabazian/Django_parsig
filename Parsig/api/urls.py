from django.urls import path
from .views import parsig_pos

urlpatterns = [
    path('pos/', parsig_pos, name='parsig_pos'),
]