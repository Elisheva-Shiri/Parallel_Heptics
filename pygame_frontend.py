import pygame
import socket
from consts import HEIGHT, PYGAME_PORT, TARGET_CYCLE_COUNT, VIRTUAL_WORLD_FPS, WIDTH
from structures import GamePacket

class PygameFrontEnd:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, server_address: str ="localhost", server_port: int = PYGAME_PORT):
        self._width = width
        self._height = height
        self._server_address=server_address
        self._server_port=server_port
        self._virtual_world_fps = VIRTUAL_WORLD_FPS

        self._running = False

        # Initialize socket to receive backend information
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind((server_address, server_port))
        
        # Initialize pygame for visualization and keyboard input
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont('Arial', 30)
        self.screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption("Hand Tracking Visualization")

    
    def start(self):
        clock = pygame.time.Clock()
        self._running = True

        while self._running:
            self._draw_visualization()
            clock.tick(self._virtual_world_fps)
        

    def _draw_visualization(self):
        """Draw fingers and virtual object visualization"""
        # TODO - Add red rectangle signaling the middle
        for event in pygame.event.get():
            if not self._handle_pygame_events(event):
                return

        self.screen.fill((0, 0, 0))

        data, _ = self._udp_socket.recvfrom(4096) # ? Is this enough for all the data?
        # Parse the network data into GamePacket structure
        packet = GamePacket.model_validate_json(data)
        
        # Draw all fingers as circles
        for finger_position in packet.landmarks:
            pygame.draw.circle(self.screen, (255, 0, 0),
                            (int(finger_position.x * self._width), int(finger_position.z * self._height)), 5)
        
        # Draw virtual object
        # TODO: Default color gray, once pinched set color based on object IDX (first Orange, then Blue)
        tracking_obj = packet.trackingObject
        color = (255, 165, 0) if tracking_obj.isPinched else (128, 128, 128)
        size = packet.trackingObject.size
        rect = pygame.Rect(
            int(tracking_obj.x * self._width - size / 2),
            int(tracking_obj.z * self._height - size / 2),
            int(size),
            int(size)
        )
        pygame.draw.rect(self.screen, color, rect)
        
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        bar_x = (self._width - bar_width) // 2
        bar_y = self._height - 40
        
        # Background bar
        pygame.draw.rect(self.screen, (64, 64, 64),
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Progress fill
        fill_width = int(bar_width * tracking_obj.progress)
        if fill_width > 0:
            pygame.draw.rect(self.screen, (0, 255, 0),
                            (bar_x, bar_y, fill_width, bar_height))
            
        # Draw movement counter
        counter_text = self._font.render(f"{tracking_obj.cycleCount}/{TARGET_CYCLE_COUNT}", True, (255, 255, 255))
        self.screen.blit(counter_text, (self._width - 50, self._height - 40))

        pygame.display.flip()

    def _handle_pygame_events(self, event: pygame.event.Event) -> bool:
        """ Handle pygame events and returns if the program should continue running"""
        match event.type:
            # case pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         self.toggle_pinch()
            case pygame.QUIT:
                self._running = False
                pygame.quit()
                return False
        return True
    
if __name__ == "__main__":
    frontend = PygameFrontEnd()
    frontend.start()
    