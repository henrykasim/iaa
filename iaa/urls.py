"""iaa URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from rest_framework import routers
from django.conf import settings
from core import views

router = routers.DefaultRouter()

urlpatterns = [
    #path('admin/', admin.site.urls),
    url(r'^', include(router.urls)),
    
    # Predictive Model
    url(r'^runPM/$', views.RunPredictiveModelAPI.as_view()),
    url(r'^pmdata/(?P<uid>\w+)/$', views.PMDataAPIView.as_view()),
    url(r'^pm/(?P<uid>\w+)/$', views.PMAPIView.as_view()),

    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]
