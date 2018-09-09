import pygame
import NeuralNetworks

class Drawing:
    #All units of drawing are given by size compared to screen*
    #For example a circle with radius of 0.1 will have a radius of 10% of the screen
    #*Outline of shapes are still given by pixels
    def __init__(self, resolution, autoUpdate = False):
        self.PyGame = pygame
        self.PyGame.init()
        self.Resolution = resolution
        self.AutoUpdate = autoUpdate
        self.Display = pygame.display.set_mode(resolution)

    def TryAutoUpdate(self):
        if self.AutoUpdate == True:
            self.Update()

    def Update(self):
        self.PyGame.display.update()

    def DrawCircle(self, color, position, radius, width = 0):
        self.PyGame.draw.circle(self.Display,color,(int(self.Resolution[0]*position[0]),int(self.Resolution[1]*position[1])),int(radius*self.Resolution[1]),width)
        self.TryAutoUpdate()

    def DrawLine(self, color, a_pos, b_pos):
        startPos = (int(self.Resolution[0]*a_pos[0]),int(self.Resolution[1]*a_pos[1]))
        endPos = (int(self.Resolution[0]*b_pos[0]),int(self.Resolution[1]*b_pos[1]))
        self.PyGame.draw.aaline(self.Display, color, startPos, endPos)
        self.TryAutoUpdate()

    def DrawRectangle(self, color, rect, width = 0):
        tempRec = (int(rect[0]*self.Resolution[0]),int(rect[1]*self.Resolution[1]),int(rect[2]*self.Resolution[0]),int(rect[3]*self.Resolution[1]))
        self.PyGame.draw.rect(self.Display,color,tempRec,width)
        self.TryAutoUpdate()

    def Fill(self,color):
        self.Display.fill(color)
        self.TryAutoUpdate()

    def Close():
        self.PyGame.quit()

    def Color(self, r, g, b):
        return self.PyGame.Color(r,g,b,0)

    def DrawNeuralNetwork(self,neuralNetwork):

        #Disable autoupdate so we dont start changing the screen before all the network is drawn,
        #this way it all updates at once
        savedAutoUpdate = self.AutoUpdate
        self.AutoUpdate = False

        White = self.Color(255,255,255)
        Black = self.Color(0,0,0)

        #Space from edge of screen we start to draw on
        start = 0.1

        #We give the neurons space equal to 70% of screen(middle of neuron will be on 70) does not include output lines
        space = 0.7

        #The map will store number of neurons/inputs per layer (inputs are layer 0)
        map = []

        #Pos will store the position of each input/neuron in the network
        pos = []

        #Will store the maximum amount of neurons per a layer so we can later decide on a size for the neurons
        maxLayer = 0

        #Make an array of the weights for easy access, will later be used to decide color of connecting lines
        weights = []
        for i in neuralNetwork.NeuronLayers:
            weights.append(i.Weights)

        #Find number of neurons per layer and store the largest number of neurons in a layer
        for i in neuralNetwork.NeuronLayers:
            map.append(i.NumberOfInputs)
            if i.NumberOfInputs > maxLayer:
                maxLayer = i.NumberOfInputs

        #using the max number of neuron per layer we decide the radius of the neuron
        neuronRadius = 0.8/(maxLayer*4)

        #Map the amount of neurons of the last layer
        map.append(neuralNetwork.NeuronLayers[len(neuralNetwork.NeuronLayers)-1].NumberOfNeurons)

        #This is the number of layers
        length = len(map)

        #Calculating each neurons position on display
        #Loop through each layer of network including inputs
        for i in range(length):
            pos.append([])
            #Loop through every neuron/inputs of layer i
            for j in range(map[i]):
                # Calculate space from left for the layer, Calculate space from top for each neuron/input
                t = (((space/(length-1))*i)+start,((space/(map[i]-1)*j)+start))
                pos[i].append(t)
        
        #Draw Lines from each layer to the layer after it and color the lines depending on weight
        #Loop through each layer execpt the last (because we draw from each layer to the next)
        for i in range(len(pos)-1):
            #Lopp through each neuron in layer i (we draw from this layer)
            for j in range(len(pos[i])):
                #Loop through each neuron in layer i+1 (we draw to this layer)
                for k in range(len(pos[i+1])):
                    #This gives us the weight index of the weight corresponding to the line we are going to draw
                    iW = j+1+k*(map[i]+1)
                    self.DrawLine(self.WeightColor(weights[i][iW]),pos[i][j],pos[i+1][k])

        #Draw bias lines
        #Loop through all layers with bias (all execpt inputs)
        for i in range(1,len(pos)):
            #Loop through each neuron in layer i
            for j in range(len(pos[i])):
                #This gives us the weight index of the weight corresponding to the line we are going to draw (the bias weight)
                iW = j*(map[i-1]+1)
                #Since the bias dosen't connect to other layers we arbitreirly decide to draw it from above
                #neuron by 2 times the radius of the neuron
                self.DrawLine(self.WeightColor(weights[i-1][iW]),pos[i][j],(pos[i][j][0],(pos[i][j][1]-neuronRadius*2)))

        #Draw output lines
        #Loop through each neuron in the last layer
        for i in range(len(pos[len(pos)-1])):
            #Since these lines don't connect to other layers we arbitreirly decide the lines should 
            #extend 2 times the radius of the neuron to the right of the output neuron layer
            t = (pos[len(pos)-1][i][0]+ neuronRadius * 2,pos[len(pos)-1][i][1])
            self.DrawLine(Black,pos[len(pos)-1][i],t)

        #Draw each neuron as a black circle (We draw them last so the lines won't be on top
        #of the circles)
        #Loop through each layer excluding inputs since we don't draw them as neurons
        for i in range(1,len(pos)):
            #Loop through each neuron in layer i
            for j in range(len(pos[i])):
                self.DrawCircle(Black,pos[i][j],neuronRadius)

        self.AutoUpdate = savedAutoUpdate
        self.TryAutoUpdate()

    def WeightColor(self,w):
        #Decide the line color based on the equivelant weight
        #if the weight is positive it will be green with max green being a weight of 1
        if w>=0:
            return self.Color(0,int(255*w),0)
        #if the weight is negative it will be red with max red being a weight of -1
        if w<0:
            return self.Color(int(255*-w),0,0)

    def CheckForQuitEvent(self):
        #Check if the user closed the pygame window
        for event in self.PyGame.event.get():
            if event.type == self.PyGame.QUIT:
                return True
            else:
                return False


        

