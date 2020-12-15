import pygame
import imageio
from vae_example_conv import *
import random


NMB_HIDDEN = 16
DISTANCE = NMB_HIDDEN*50 + 80
net = VAE()
checkpoint = torch.load('checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
net.eval()

def loadImg(location='test.png'):
    return torch.Tensor(imageio.imread(location)[:,:,:3]).permute(2,0,1).unsqueeze(0) / 255.

def toImg(emb):
    return net.decode(emb)[0].permute(1,2,0).detach()

def loadData(x):
    img = torch.Tensor(imageio.imread(x))
    dset = torch.zeros(35*35, 32,32,3)
    tel = 0
    for y in range(0,35*32,32):
        for x in range(0,35*32,32):
            dset[tel] = img[y:y+32, x:x+32]
            tel += 1
    return dset.permute((0,3,1,2))[:-52]/255.

imagesTrain = loadData('dataset.png')
    
class Bar:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.moving = False
    def update(self, dp):
        if self.moving:
            self.y = pygame.mouse.get_pos()[1]
        pygame.draw.rect(dp, (0,0,0), [self.x-20, self.y-8, 40, 16])
        pygame.draw.rect(dp, (100,100,100), [self.x-16, self.y-5, 32, 10])
        pos = -self.getRel()
        if pos < -3.5: self.y = (3.5*20)+200
        if pos > 3.5:  self.y  =  200-(3.5*20)
        draw(dp, '%.2f' % (-self.getRel()),self.x-25, 350 , (0,255,0), (0,0,255))
    def checkPressed(self, mousePosition):
        if self.x-20 < mousePosition[0] < self.x+20 and self.y-6 < mousePosition[1] < self.y+6:
            self.moving = True
    def unclick(self):
        self.moving = False
    def getRel(self):
        return ((self.y -200) / 20.)

class RandomButton():
    def __init__(self):
        self.isPressed = False
    def update(self, dp):
        draw(dp, 'Click for new Random Image', DISTANCE-20, 30, (0,255,0), (0,0,255))
    def checkPressed(self, mp):
        return DISTANCE-15 < mp[0] < DISTANCE+300 and 20 < mp[1] < 80

def draw( dp, text, x,y, c1= (255,20,20), c2 = (255,255,255)):
    text = font.render(text, True, c1, c2) 
    rect = text.get_rect() 
    rect.topleft = (x, y)
    dp.blit(text, rect)     

def load():
    img = (imagesTrain[random.randint(0, len(imagesTrain)-1)]*255).permute(1,2,0).numpy()
    imageio.imsave('test.png', img.reshape((32,32,3)))
    img  = pygame.transform.scale(pygame.image.load('test.png'), (100,100))
    testImage(net, 0)
    reconstruction = pygame.transform.scale(pygame.image.load('test_reconstruction_0.png'), (100,100))
    torchImg = loadImg()
    starty = [float(i)for i in net.encode(torchImg)[0].view(-1)]
    bars = [Bar(50*(i+1), (20*starty[i])+200) for i in range(len(starty))]
    return img, reconstruction,bars

def testImage(net, i=0):
    img = loadImg()
    with torch.no_grad():
        prediction = net.decode(net.encode(img)[0])[0]
        prediction = (prediction.permute(1,2,0).numpy()*255).astype('uint8')
        imageio.imsave('test_reconstruction_%i.png' % i, prediction)

if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font('freesansbold.ttf', 20)
    gameDisplay = pygame.display.set_mode((DISTANCE+300, 400))
    pygame.display.set_caption('VAE example')
    clock = pygame.time.Clock()
    
    img, reconstruction,bars = load()
    randomButton = RandomButton()
    reload = False
    
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONUP:
                [bar.unclick() for bar in bars]
                if randomButton.checkPressed(pygame.mouse.get_pos()):
                    img, reconstruction,bars = load()
                reload =  False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mp = pygame.mouse.get_pos()
                [bar.checkPressed(mp)for bar in bars]
                reload =  True
        if reload:
            decon = torch.Tensor([[(i.y-200)/20. for i in bars]])
            prediction = net.decode(decon).detach()[0].permute(1,2,0)
            prediction = (prediction.numpy()*255).astype('uint8')
            imageio.imsave('test_reconstruction_0.png', prediction)
            reconstruction = pygame.transform.scale(pygame.image.load('test_reconstruction_0.png'), (100,100))
        gameDisplay.fill((255,255,255))
        [bar.update(gameDisplay) for bar in bars]
        randomButton.update(gameDisplay)
        gameDisplay.blit(img, (DISTANCE,90))
        gameDisplay.blit(reconstruction, (DISTANCE,220))
        draw(gameDisplay, '<-- Input', DISTANCE+110, 130)
        draw(gameDisplay, '<-- Reconstruction', DISTANCE+110, 260)
        pygame.display.update()
        clock.tick(30)
        
