import pygame
import random
import socket
import sys
from array import array
from consts import BACKEND_PORT, TOP_HEIGHT, PYGAME_PORT, FRONTEND_FPS, TOP_WIDTH
from structures import ExperimentControl, ExperimentPacket, ExperimentState, FingerPosition, QuestionInput, TrackingObject

FIRST_COLOR = (255, 165, 0)  # Orange
SECOND_COLOR = (0, 0, 255)  # Blue
WHITE_NOISE_SAMPLE_RATE = 44_100
WHITE_NOISE_SECONDS = 2.0
WHITE_NOISE_VOLUME = 0.15


def _make_white_noise_buffer(
    sample_rate: int = WHITE_NOISE_SAMPLE_RATE,
    seconds: float = WHITE_NOISE_SECONDS,
    amplitude: float = WHITE_NOISE_VOLUME,
    seed: int | None = None,
) -> bytes:
    """Generate little-endian signed 16-bit mono white-noise PCM."""
    sample_count = max(1, int(sample_rate * seconds))
    peak = max(0, min(32767, int(32767 * amplitude)))
    rng = random.Random(seed)
    samples = array("h", (rng.randint(-peak, peak) for _ in range(sample_count)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


class _WhiteNoiseLoop:
    """Best-effort looping white noise for the pygame/computer frontend."""

    def __init__(self, enabled: bool = False, volume: float = WHITE_NOISE_VOLUME):
        self._enabled = enabled
        self._volume = max(0.0, min(1.0, volume))
        self._sound: pygame.mixer.Sound | None = None
        self._channel: pygame.mixer.Channel | None = None

    def pre_init_mixer(self) -> None:
        pygame.mixer.pre_init(WHITE_NOISE_SAMPLE_RATE, -16, 1, 2048)

    def start(self) -> None:
        if not self._enabled or self._channel is not None:
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(WHITE_NOISE_SAMPLE_RATE, -16, 1, 2048)
            self._sound = pygame.mixer.Sound(
                buffer=_make_white_noise_buffer(amplitude=0.8, seed=0)
            )
            self._sound.set_volume(self._volume)
            self._channel = self._sound.play(loops=-1)
        except pygame.error as ex:
            # Keep the experiment usable on machines without an audio device.
            print(f"White noise disabled: {ex}")
            self._enabled = False
            self._sound = None
            self._channel = None

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled:
            self.pre_init_mixer()
            self.start()
        else:
            self.stop()

    def stop(self) -> None:
        if self._channel is not None:
            self._channel.stop()
            self._channel = None
        self._sound = None


class PygameFrontEnd:
    def __init__(
        self,
        width: int = TOP_WIDTH,
        height: int = TOP_HEIGHT,
        server_address: str = "localhost",
        frontend_port: int = PYGAME_PORT,
        backend_port: int = BACKEND_PORT,
        white_noise_volume: float = WHITE_NOISE_VOLUME,
    ):
        self._width = width
        self._height = height
        self._server_address=server_address
        self._frontend_port=frontend_port
        self._backend_port=backend_port
        self._virtual_world_fps = FRONTEND_FPS

        self._running = False
        self._left_button_timer = 0
        self._left_button_sent: bool = False
        self._right_button_timer = 0
        self._right_button_sent: bool = False
        self._button_hold_time = FRONTEND_FPS  # 1 second worth of frames (2x faster fill)
        self._white_noise = _WhiteNoiseLoop(enabled=False, volume=white_noise_volume)

        # Initialize socket to receive backend information
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._data_socket.bind((server_address, frontend_port))

        # Initialize socket to send user input to backend
        self._input_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._input_socket.connect((server_address, backend_port))
        
        # Initialize pygame for visualization and keyboard input
        self._white_noise.pre_init_mixer()
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont('Arial', 30)
        self._title_font = pygame.font.SysFont('Arial', 40)
        self.screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption("Hand Tracking Visualization")
        self._white_noise.start()

    
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
        bar_y = 40
        half_width = bar_width // 2
        outbound_progress = max(0.0, min(tracking_obj.progress, 1.0))
        return_progress = max(0.0, min(tracking_obj.returnProgress, 1.0))
        
        # Background bar
        pygame.draw.rect(self.screen, (64, 64, 64),
                        (bar_x, bar_y, bar_width, bar_height))
        
        # First half: center -> side (light green)
        first_fill_width = int(half_width * outbound_progress)
        if first_fill_width > 0:
            pygame.draw.rect(self.screen, (144, 238, 144),
                            (bar_x, bar_y, first_fill_width, bar_height))

        # Second half: side -> center (green)
        second_fill_width = int(half_width * return_progress)
        if second_fill_width > 0:
            pygame.draw.rect(self.screen, (0, 200, 0),
                            (bar_x + half_width, bar_y, second_fill_width, bar_height))
            
        # Draw movement counter
        counter_text = self._font.render(f"{tracking_obj.cycleCount}/{tracking_obj.targetCycleCount}", True, (255, 255, 255))
        self.screen.blit(counter_text, (self._width - 50, 40))

    def _draw_question(self, landmarks: list[FingerPosition]) -> None:
        # Draw title
        title = self._title_font.render("Which object is stiffer?", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self._width/2, 50))
        self.screen.blit(title, title_rect)

        # Draw buttons
        button_width = 100
        button_height = 100
        left_button = pygame.Rect(90, self._height/2 - button_height/2, button_width, button_height)
        right_button = pygame.Rect(self._width - 190, self._height/2 - button_height/2, button_width, button_height)
        
        pygame.draw.rect(self.screen, FIRST_COLOR, left_button)
        pygame.draw.rect(self.screen, SECOND_COLOR, right_button)

        # Check if any fingers are touching buttons
        left_touched = False
        right_touched = False
        
        for finger_position in landmarks:
            finger_pos = (int(finger_position.x * self._width), int(finger_position.z * self._height))
            
            if left_button.collidepoint(finger_pos):
                left_touched = True
            if right_button.collidepoint(finger_pos):
                right_touched = True

        # Update button states based on touches
        if left_touched:
            self._left_button_timer += 1
            if self._left_button_timer >= self._button_hold_time and not self._left_button_sent:
                self._left_button_timer = self._button_hold_time  # limit the timer to the hold time
                self._input_socket.sendall(
                    ExperimentControl(questionInput=QuestionInput.LEFT.value).model_dump_json().encode()
                )
                self._left_button_sent = True
        else:
            self._left_button_timer = 0
            self._left_button_sent = False

        if right_touched:
            self._right_button_timer += 1
            if self._right_button_timer >= self._button_hold_time and not self._right_button_sent:
                self._right_button_timer = self._button_hold_time  # limit the timer to the hold time
                self._input_socket.sendall(
                    ExperimentControl(questionInput=QuestionInput.RIGHT.value).model_dump_json().encode()
                )
                self._right_button_sent = True
        else:
            self._right_button_timer = 0
            self._right_button_sent = False

        # Draw progress bars for button holds
        if self._left_button_timer > 0:
            progress = self._left_button_timer / self._button_hold_time
            pygame.draw.rect(self.screen, (64, 64, 64), (left_button.x, left_button.bottom + 10, button_width, 10))
            pygame.draw.rect(self.screen, (0, 255, 0), (left_button.x, left_button.bottom + 10, button_width * progress, 10))

        if self._right_button_timer > 0:
            progress = self._right_button_timer / self._button_hold_time
            pygame.draw.rect(self.screen, (64, 64, 64), (right_button.x, right_button.bottom + 10, button_width, 10))
            pygame.draw.rect(self.screen, (0, 255, 0), (right_button.x, right_button.bottom + 10, button_width * progress, 10))

    def _draw_pause(self, pause_time: int):
        # Draw pause screen title
        title = self._title_font.render("Take a break!", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self._width/2, self._height/2 - 30))
        self.screen.blit(title, title_rect)

        # Draw pause screen subtitle
        subtitle = self._font.render(f"{pause_time} seconds left before moving to the next test...", True, (255, 255, 255))
        subtitle_rect = subtitle.get_rect(center=(self._width/2, self._height/2 + 10))
        self.screen.blit(subtitle, subtitle_rect)

    def _draw_break(self, elapsed_seconds: int):
        title = self._title_font.render("Break — press Enter on the keyboard", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self._width/2, self._height/2 - 50))
        self.screen.blit(title, title_rect)

        subtitle = self._font.render(f"Break so far: {elapsed_seconds} s", True, (255, 255, 255))
        subtitle_rect = subtitle.get_rect(center=(self._width/2, self._height/2 + 10))
        self.screen.blit(subtitle, subtitle_rect)

        instr = self._font.render("Switch fingers on the screen", True, (255, 255, 255))
        instr_rect = instr.get_rect(center=(self._width/2, self._height/2 + 50))
        self.screen.blit(instr, instr_rect)

    def _draw_end(self):
        # Draw end screen title
        title = self._title_font.render("Thank you for participating!", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self._width/2, self._height/2 - 30))
        self.screen.blit(title, title_rect)

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
        self._white_noise.set_enabled(packet.playWhiteNoise)
        
        # Draw all fingers as circles
        for finger_position in packet.landmarks:
            pygame.draw.circle(self.screen, (211, 211, 211),
                            (int(finger_position.x * self._width), int(finger_position.z * self._height)), 5)
        
        match packet.stateData.state:
            case ExperimentState.COMPARISON.value:
                self._draw_comparison(packet.trackingObject)

            case ExperimentState.QUESTION.value:
                self._draw_question(packet.landmarks)

            case ExperimentState.PAUSE.value:
                self._draw_pause(packet.stateData.pauseTime)

            case ExperimentState.BREAK.value:
                self._draw_break(packet.stateData.pauseTime)

            case ExperimentState.END.value:
                self._draw_end()

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
                self._white_noise.stop()
                pygame.quit()
                return False
        return True


if __name__ == "__main__":
    frontend = PygameFrontEnd()
    frontend.start()    
