import pygame
import random
import neat
from sys import exit

# set screen and font
pygame.init()
screen = pygame.display.set_mode((500, 900))
pygame.display.set_caption("AI Flappy Bird")
font = pygame.font.Font("fonts/Minecraft.ttf", 50)


class Bird:

    def __init__(self):
        # bird images
        birdFlap1 = pygame.image.load("images/bird1.png").convert_alpha()
        birdFlap2 = pygame.image.load("images/bird2.png").convert_alpha()
        self.birdFlaps = [birdFlap1, birdFlap2]
        self.img = self.birdFlaps[0]
        self.imgIndex = 0
        # set initial position and gravity
        self.x = 230
        self.y = 350
        self.gravity = 0

    def jump(self):
        self.gravity = -15

    def applyGravity(self):
        self.gravity += 1
        self.y += self.gravity

    def animate(self, screen):
        # rotate through images each frame to get animation effect
        self.imgIndex += 0.1
        if self.imgIndex >= len(self.birdFlaps):
            self.imgIndex = 0
        self.img = self.birdFlaps[int(self.imgIndex)]
        screen.blit(self.img, (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe():

    def __init__(self, x):
        # pipe images
        self.pipeTop = pygame.image.load("images/tPipe.png").convert_alpha()
        self.pipeBottom = pygame.image.load("images/bPipe.png").convert_alpha()
        # set pipe positions
        self.x = x
        self.y = random.randrange(50, 500)
        self.top = self.y - self.pipeTop.get_height()
        self.bottom = self.y + 200
        # boolean to keep track of whether bird has passed the pipe
        self.passed = False

    def animate(self):
        self.x -= 5

    def draw(self, screen):
        screen.blit(self.pipeTop, (self.x, self.top))
        screen.blit(self.pipeBottom, (self.x, self.bottom))

    def collide(self, bird):
        # check if bird has collided with pipe
        birdMask = bird.get_mask()
        topMask = pygame.mask.from_surface(self.pipeTop)
        bottomMask = pygame.mask.from_surface(self.pipeBottom)
        # general apprach inspired by "HKH" from stack overflow
        topOffset = (self.x - bird.x, self.top - bird.y)
        bottomOffset = (self.x - bird.x, self.bottom - bird.y)
        if birdMask.overlap(bottomMask, bottomOffset) or birdMask.overlap(topMask, topOffset):
            return True
        return False


class Ground:

    def __init__(self, y):
        self.y = y
        self.x = 0

    def animate(self):
        self.x -= 4
        if self.x <= -500:
            self.x = 0

    def draw(self, screen):
        ground = pygame.image.load("images/ground.png").convert_alpha()
        screen.blit(ground, (self.x, self.y))


def draw_window(screen, birds, pipes, ground, score, gen):
    # draw background, birds, pipes, and ground
    backgroundImage = pygame.image.load("images/bg.png").convert_alpha()
    screen.blit(backgroundImage, (0, 0))
    ground.draw(screen)
    for pipe in pipes:
        pipe.draw(screen)
    for bird in birds:
        bird.animate(screen)

    # display score and generation
    scoreSurface = font.render(f"SCORE: {score}", False, ("White"))
    screen.blit(scoreSurface, (10, 60))
    genSurface = font.render(f"GEN: {gen-1}", False, "White")
    screen.blit(genSurface, (10, 10))
    pygame.display.update()


# function ran for each new generation
def eval_genomes(genomes, config):
    # set score to 0 and increment generation
    global screen, gen
    clock = pygame.time.Clock()
    gen += 1
    score = 0

    # create genomes with fitness 0 and add it genomeList, then add them to the neural network, and append the approriate birds to birds list
    neuralNetworks = []
    birds = []
    genomeList = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        neuralNetworks.append(net)
        birds.append(Bird())
        genomeList.append(genome)

    # instanciate ground and pipes
    ground = Ground(700)
    pipes = [Pipe(700)]

    # while at least one bird from a generation is still alive
    while len(birds) > 0:
        # give player option to quit game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        # set game to 30 FPS and animate ground
        clock.tick(30)
        ground.animate()

        # determine if bird has passed through pipe if so update pipe
        pipeIndex = 0
        if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipeTop.get_width():
            pipeIndex = 1

        # give each bird a fitness of 0.1 for each frame it stays alive
        for i, bird in enumerate(birds):
            genomeList[i].fitness += 0.1
            bird.applyGravity()

            # send bird location, top pipe location, and bottom pipe location to determine whether to jump or not
            output = neuralNetworks[i].activate(
                (bird.y, abs(bird.y - pipes[pipeIndex].y), abs(bird.y - pipes[pipeIndex].bottom)))

            # since we're using the tanh function(between -1 and 1), if over 0.5 we jump
            if output[0] > 0.5:
                bird.jump()

            # eliminate bird if goes off screen or touches floor
            if bird.y + bird.img.get_height() - 10 >= 700 or bird.y < -50:
                neuralNetworks.pop(birds.index(bird))
                genomeList.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        for pipe in pipes:
            # animate pipe
            pipe.animate()

            # if collision occurs remove bird
            for bird in birds:
                if pipe.collide(bird):
                    neuralNetworks.pop(birds.index(bird))
                    genomeList.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            # generate new pipe once pipe is passed and increment score
            if not pipe.passed and pipe.x < bird.x-30:
                pipe.passed = True
                score += 1
                pipes.append(Pipe(500))

            # remove pipe once it is off screen
            if pipe.x <= -100:
                pipes.remove(pipe)

        draw_window(screen, birds, pipes, ground, score, gen)


if __name__ == '__main__':
    # connect neat with configuration file
    config_path = 'configFile.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    # create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)
    # add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # run for up to 50 generations starting with gen 0.
    gen = 0
    population.run(eval_genomes, 50)
