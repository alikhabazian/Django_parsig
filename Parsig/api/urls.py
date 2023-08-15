from django.urls import path
from .views import parsig_pos,parsig_Seq2Seq

urlpatterns = [
    path('pos/', parsig_pos, name='parsig_pos'),
    path('seq2seq/', parsig_Seq2Seq, name='run'),
]