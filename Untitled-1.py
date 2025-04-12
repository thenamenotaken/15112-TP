from cmu_graphics import *

def onAppStart(app):
    app.width = 400
    app.height = 400

def redrawAll(app):
    drawRect(0, 0, 50, 50, fill='blue')  


runApp()
