# Water Tracker 3.0, forewarning it senses rlly bright lights as hands
import pygame
import cv2
import mediapipe as mp
import math
import random
import numpy as np

class Ripple:
    def __init__(self, x, y, growthRate=4):
        self.x = x
        self.y = y
        self.size = 0
        self.maxSize = random.randint(50, 400)
        self.growthRate = growthRate
        self.color = (random.randint(180, 255), 
                      random.randint(180, 255), 
                      random.randint(200, 255))
    
    def update(self):
        self.size += self.growthRate
        return self.size <= self.maxSize
    
    def getFadeFactor(self):
        return 1 - self.size / self.maxSize

class WaterTracker:
    def __init__(self):
        pygame.init()
        
        self.wrist = 0
        self.thumbCmc = 1      
        self.thumbMcp = 2      
        self.thumbIp = 3       
        self.thumbTip = 4
        self.indexMcp = 5
        self.indexPip = 6      
        self.indexDip = 7      
        self.indexTip = 8
        self.middleMcp = 9
        self.middlePip = 10
        self.middleDip = 11
        self.middleTip = 12
        self.ringMcp = 13
        self.ringPip = 14
        self.ringDip = 15
        self.ringTip = 16
        self.pinkyMcp = 17
        self.pinkyPip = 18
        self.pinkyDip = 19
        self.pinkyTip = 20
        #^^^for conveniency of tracking hand, if you get confused, run 
        #Hand Tracker Num Ref

        self.width = 640
        self.height = 480
        self.canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Water Tracker 3.0")
        #Just a heads up Yulissa, anytime snake case is used, it's usually
        #a function of pygame, like .set_mode creates the canvas we use and
        #.set_caption names that canvas
        
        self.frameRate = 40
        self.clock = pygame.time.Clock()
        #this is EQUIV to StepsPerSecond ^^^
        

        # Hand tracking stuff internet helped with
        self.mpHands = mp.solutions.hands
        # Updated to use the correct parameters for MediaPipe 0.8.9 and above
        self.hands = self.mpHands.Hands(
            max_num_hands=2,  
            min_detection_confidence=0.7,  
            min_tracking_confidence=0.6    
        )
        
        # Camera Setup
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  

        self.ripples = []
        self.pinchThreshold = 40
        
        self.font = pygame.font.SysFont(None, 24)
        #size 24 default system font

    
        
    def update(self):
        #This below returns a bool for if camera frame was captured anddd
        #the NumPy of the next frame
        success, frame = self.capture.read()
        if not success:
            self.statusMessage = "Camera not available"
            return
            
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        
        # CV does BGR for some reason so I'm just converting to RGB here for 
        #MP
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgbFrame)
        
        # we save frame to draw water
        self.currentFrame = rgbFrame
        
        # Process hand landmarks
        handLandmarks = [] #for diff hands
        if results.multi_hand_landmarks: #if hand spawns
            for handLandmark in results.multi_hand_landmarks:
                landmarks = [] #for pts on hand
                for point in handLandmark.landmark:
                    # MP returns value (0-1) so I just convert
                    x, y = int(point.x * self.width), int(point.y * self.height)
                    landmarks.append((x, y))
                handLandmarks.append(landmarks)
        
        self.processGestures(handLandmarks)
        self.updateRipples()
        
        self.statusMessage = f"Hands detected: {len(handLandmarks)}"
        
    def ptpDistance(self,x1,y1,x2,y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def processGestures(self, handsData):
        if not handsData:
            return
            
        for landmarks in handsData:
            if len(landmarks) < 21:  # MP detects 21 pts per hand
                continue
                
        
        thumbTip = landmarks[self.thumbTip]
        indexTip = landmarks[self.indexTip]
        
        distance = self.ptpDistance(thumbTip[0],thumbTip[1],indexTip[0], 
                                    indexTip[1])
        #[0] is x and [1] is y
        
        # droplet pinch effect
        if distance < self.pinchThreshold:
            midpointX = (thumbTip[0] + indexTip[0]) / 2
            midpointY = (thumbTip[1] + indexTip[1]) / 2
            
            self.createRipple(midpointX, midpointY)
        
        # full rain fist effect
        ringTip = landmarks[self.ringTip]
        thumbBase = landmarks[self.thumbCmc]
        wrist = landmarks[self.wrist]
        
        dist1 = self.ptpDistance(thumbBase[0], thumbBase[1], indexTip[0], indexTip[1]) 
        dist2 = self.ptpDistance(wrist[0], ringTip[1], wrist[0], ringTip[1]) 
        
       
        if dist1 <= 50 and dist2 <= 50:
            self.createRain()
    
    def createRipple(self, x, y, count=1):
        self.ripples.append(Ripple(x, y))
    
    def createRain(self):
        for i in range(3):
            self.createRipple(
                random.randint(0, self.width),  
                random.randint(0, self.height)  
            )
    
    def updateRipples(self):
        activeRipples = []
    
        for ripple in self.ripples:
            if ripple.update():
                activeRipples.append(ripple)
        
            self.ripples = activeRipples
            
            if len(self.ripples) > 50:
                self.ripples = self.ripples[-50:] #most recent 50 ripples
    
    def draw(self):
        # Draw webcam image if available - ENSURE IT COVERS THE FULL WINDOW
        if hasattr(self, 'currentFrame'):
            # The image is already in RGB format (converted in update method)
            frame = self.currentFrame
            
            # Create surface from numpy array - this will cover the whole canvas
            # Check if frame has the right shape before creating surface
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                
               
                self.canvas.blit(surf, (0, 0))
        
       
        self.drawRipples()
        
       
        self.drawHandTracking()
        
     
        if not hasattr(self, 'statusMessage') or "Hands detected: 0" in self.statusMessage:
            self.drawInstructions()
        
       
        statusText = self.font.render(
            f"Press Q to quit | {self.statusMessage}", True, (255, 255, 255))
        self.canvas.blit(statusText, (10, self.height - 30))  # Fixed: using self.height instead of self.canvasHeight
    
    def drawHandTracking(self):
        if hasattr(self, 'mpHands') and hasattr(self, 'currentFrame'):
           
            rgbFrame = self.currentFrame
            results = self.hands.process(rgbFrame)
            
            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    
                    thumbTip = (int(handLandmarks.landmark[self.thumbTip].x * self.width), 
                                int(handLandmarks.landmark[self.thumbTip].y * self.height))
                    indexTip = (int(handLandmarks.landmark[self.indexTip].x * self.width), 
                                int(handLandmarks.landmark[self.indexTip].y * self.height))
                    
                    pygame.draw.circle(self.canvas, (255, 0, 0), thumbTip, 8)
                    pygame.draw.circle(self.canvas, (0, 255, 0), indexTip, 8)
                    
                    pygame.draw.line(self.canvas, (255, 255, 0), thumbTip, indexTip, 2)
                    
                    # Draw distance text
                    dist = self.ptpDistance(thumbTip[0], thumbTip[1], indexTip[0], indexTip[1])
                    distText = self.font.render(f"{int(dist)}", True, (255, 255, 255))
                    midpoint = ((thumbTip[0] + indexTip[0])//2, 
                               (thumbTip[1] + indexTip[1])//2 - 15)
                    self.canvas.blit(distText, midpoint)
    
    def drawRipples(self):
        for ripple in self.ripples:
            # Calculate opacit
            fadeFactor = ripple.getFadeFactor()
            
            # Outer circle
            outerColor = (int(65 + (255-65)*fadeFactor), 
                        int(130 + (255-130)*fadeFactor), 
                        int(117 + (255-117)*fadeFactor))
            
            pygame.draw.circle(
                self.canvas, 
                outerColor, 
                (int(ripple.x), int(ripple.y)), 
                int(ripple.size/2), 
                3
            )
            
            # Inner circle
            if len(self.ripples) < 30:  # Only draw inner circles if not too many ripples
                innerColor = (int(241 + (255-241)*fadeFactor), 
                            int(247 + (255-247)*fadeFactor), 
                            int(240 + (255-240)*fadeFactor))
                            
                pygame.draw.circle(
                    self.canvas, 
                    innerColor, 
                    (int(ripple.x), int(ripple.y)), 
                    int(ripple.size * 0.4), 
                    4
                )
    
    def drawInstructions(self):
        """Draw instructions for the user"""
        text1 = self.font.render(
            "Pinch your thumb and index finger together to create ripples", 
            True, (255, 255, 255))
        text2 = self.font.render(
            "Make a fist to create rain effect", 
            True, (255, 255, 255))
            
        self.canvas.blit(text1, (self.width//2 - text1.get_width()//2, 30))  # Fixed: using self.width instead of self.canvasWidth
        self.canvas.blit(text2, (self.width//2 - text2.get_width()//2, 60))  # Fixed: using self.width instead of self.canvasWidth
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            self.update()
        
            self.draw()
            
            pygame.display.flip()
            self.clock.tick(self.frameRate)
        
        # Ngl this part here is AI code, still tyring to figure it out
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        if hasattr(self, 'hands') and self.hands is not None:
            self.hands.close()
        pygame.quit()

if __name__ == "__main__":
    app = WaterTracker()
    app.run()