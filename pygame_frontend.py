import pygame
import socket
from consts import BACKEND_PORT, HEIGHT, PYGAME_PORT, TARGET_CYCLE_COUNT, VIRTUAL_WORLD_FPS, WIDTH
from structures import ExperimentControl, ExperimentPacket, ExperimentState, FingerPosition, QuestionInput, TrackingObject

FIRST_COLOR = (255, 165, 0)  # Orange
SECOND_COLOR = (0, 0, 255)  # Blue

class PygameFrontEnd:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, server_address: str ="localhost", frontend_port: int = PYGAME_PORT, backend_port: int = BACKEND_PORT):
        self._width = width
        self._height = height
        self._server_address=server_address
        self._frontend_port=frontend_port
        self._backend_port=backend_port
        self._virtual_world_fps = VIRTUAL_WORLD_FPS

        self._running = False
        self._left_button_timer = 0
        self._right_button_timer = 0
        self._button_hold_time = 2 * VIRTUAL_WORLD_FPS  # 2 seconds worth of frames

        # Initialize socket to receive backend information
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._data_socket.bind((server_address, frontend_port))

        # Initialize socket to send user input to backend
        # self._input_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self._input_socket.connect((server_address, backend_port))
        
        # Initialize pygame for visualization and keyboard input
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont('Arial', 30)
        self._title_font = pygame.font.SysFont('Arial', 40)
        self.screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption("Hand Tracking Visualization")

    
    def start(self):
        clock = pygame.time.Clock()
        self._running = True

        while self._running:
            self._draw_visualization()
            clock.tick(self._virtual_world_fps)

    def _draw_comparison(self, tracking_obj: TrackingObject) -> None:
        # Draw virtual object
        color = (FIRST_COLOR if tracking_obj.pairIndex == 0 else SECOND_COLOR) if tracking_obj.isPinched else (128, 128, 128)
        size = tracking_obj.size
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

    def _draw_question(self, landmarks: list[FingerPosition]) -> None:
        # Draw title
        title = self._title_font.render("Which object is stiffer?", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self._width/2, 50))
        self.screen.blit(title, title_rect)

        # Draw buttons
        button_width = 100
        button_height = 100
        left_button = pygame.Rect(50, self._height/2 - button_height/2, button_width, button_height)
        right_button = pygame.Rect(self._width - 150, self._height/2 - button_height/2, button_width, button_height)
        
        pygame.draw.rect(self.screen, FIRST_COLOR, left_button)
        pygame.draw.rect(self.screen, SECOND_COLOR, right_button)

        # Check if any fingers are touching buttons
        for finger_position in landmarks:
            finger_pos = (int(finger_position.x * self._width), int(finger_position.z * self._height))
            
            if left_button.collidepoint(finger_pos):
                self._left_button_timer += 1
                # if self._left_button_timer >= self._button_hold_time:
                #     # ! BUG: will send data to backend over and over even if the button is not released
                #     self._input_socket.sendall(
                #         ExperimentControl(questionInput=QuestionInput.LEFT.value).model_dump_json().encode()
                #     )
            else:
                self._left_button_timer = 0

            if right_button.collidepoint(finger_pos):
                self._right_button_timer += 1
                # if self._right_button_timer >= self._button_hold_time:
                #     # ! BUG: will send data to backend over and over even if the button is not released
                #     self._input_socket.sendall(
                #         ExperimentControl(questionInput=QuestionInput.RIGHT.value).model_dump_json().encode()
                #     )
            else:
                self._right_button_timer = 0

        # Draw progress bars for button holds
        if self._left_button_timer > 0:
            progress = self._left_button_timer / self._button_hold_time
            pygame.draw.rect(self.screen, (64, 64, 64), (50, self._height/2 + 60, button_width, 10))
            pygame.draw.rect(self.screen, (0, 255, 0), (50, self._height/2 + 60, button_width * progress, 10))

        if self._right_button_timer > 0:
            progress = self._right_button_timer / self._button_hold_time
            pygame.draw.rect(self.screen, (64, 64, 64), (self._width - 150, self._height/2 + 60, button_width, 10))
            pygame.draw.rect(self.screen, (0, 255, 0), (self._width - 150, self._height/2 + 60, button_width * progress, 10))

    def _draw_visualization(self):
        """Draw fingers and virtual object visualization"""
        # TODO - Add red rectangle signaling the middle
        for event in pygame.event.get():
            if not self._handle_pygame_events(event):
                return

        self.screen.fill((0, 0, 0))

        data, _ = self._data_socket.recvfrom(4096) # ? Is this enough for all the data?
        # Parse the network data into ExperimentPacket structure
        packet = ExperimentPacket.model_validate_json(data)
        
        # Draw all fingers as circles
        for finger_position in packet.landmarks:
            pygame.draw.circle(self.screen, (255, 0, 0),
                            (int(finger_position.x * self._width), int(finger_position.z * self._height)), 5)
        
        match packet.state:
            case ExperimentState.COMPARISON.value:
                self._draw_comparison(packet.trackingObject)

            case ExperimentState.QUESTION.value:
                self._draw_question(packet.landmarks)

            # TODO - Add Pause>End>Start screens

            case _:
                pass

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
