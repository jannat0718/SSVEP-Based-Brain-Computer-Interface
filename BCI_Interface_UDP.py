##Latest ## Final Interface logic with session_plan and handelling: 1/2/3/4 to select shape, space/enter to control size
import pygame
import socket
import time
import numpy as np

# --- UDP socket setup for markers ---
UDP_IP = "127.0.0.1"  # Replace with target IP if needed
UDP_PORT = 1000        # Replace with your desired port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_trigger(marker):
    """Send marker via UDP socket"""
    try:
        sock.sendto(str(marker).encode(), (UDP_IP, UDP_PORT))
        print(f"Marker sent: {marker}")
    except Exception as e:
        print(f"Error sending marker: {e}")

# Alias to match original code
event_marker = send_trigger

# --- Keep all the original pygame setup code unchanged ---
pygame.init()
screen_width, screen_height = 1200, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("BCI Drawing Interface")

# Colors, fonts, positions (unchanged)
WHITE, BLACK = (255,255,255), (0,0,0)
GRAY, RED, GREEN, YELLOW, BLUE = (100,100,100), (255,0,0), (0,255,0), (255,255,0), (0,0,255)

circle_pos, rect_pos, triangle_pos, do_nothing_pos = (150,100),(150,280),(150,460),(150,640)
canvas_x, canvas_y, canvas_w, canvas_h = 500, 100, 500, 600

font = pygame.font.Font(None,36)
small_font = pygame.font.Font(None,24)

# Flicker info with key mappings (unchanged)
flicker_info = {
    "Circle":    {"color":RED,   "pos":circle_pos,   "hz":10,"last":time.time(),"on":True,"key":pygame.K_1},
    "Rectangle": {"color":BLUE,  "pos":rect_pos,     "hz":12,"last":time.time(),"on":True,"key":pygame.K_2},
    "Triangle":  {"color":GREEN, "pos":triangle_pos, "hz":14,"last":time.time(),"on":True,"key":pygame.K_3},
    "Do Nothing":{"color":WHITE, "pos":do_nothing_pos,"hz":16,"last":time.time(),"on":True,"key":pygame.K_4},
}

key_to_shape_map = {
    pygame.K_1:"Circle",     pygame.K_KP1:"Circle",
    pygame.K_2:"Rectangle",  pygame.K_KP2:"Rectangle",
    pygame.K_3:"Triangle",   pygame.K_KP3:"Triangle",
    pygame.K_4:"Do Nothing", pygame.K_KP4:"Do Nothing",
}
SPACE_KEYS = {pygame.K_SPACE}
ENTER_KEYS = {pygame.K_RETURN, pygame.K_KP_ENTER}

selected_shape = None
last_placed_shape, last_placed_size = None, None
focus_val, focus_dir = 0,1

def interpolate(val):
    r=int(RED[0]+(GREEN[0]-RED[0])*val/100)
    g=int(RED[1]+(GREEN[1]-RED[1])*val/100)
    b=int(RED[2]+(GREEN[2]-RED[2])*val/100)
    return r,g,b

def draw_flicker(phase):
    now=time.time()
    for name,info in flicker_info.items():
        col=GRAY
        if phase=="Selection_Phase":
            if now-info["last"]>=0.5/info["hz"]:
                info["on"]=not info["on"]; info["last"]=now
            col=info["color"] if info["on"] else GRAY
        elif phase in ["Shape Selected","Static_Dot","Size selected"]:
            col=info["color"] if selected_shape==name else GRAY
        if name=="Circle":
            pygame.draw.circle(screen,col,info["pos"],50)
            screen.blit(small_font.render("Circle",True,BLACK),
                        small_font.render("Circle",True,BLACK).get_rect(center=info["pos"]))
        elif name=="Rectangle":
            pygame.draw.rect(screen,col,(info["pos"][0]-50,info["pos"][1]-30,100,60))
            screen.blit(small_font.render("Rectangle",True,BLACK),
                        small_font.render("Rectangle",True,BLACK).get_rect(center=info["pos"]))
        elif name=="Triangle":
            pts=[(info["pos"][0],info["pos"][1]-50),
                 (info["pos"][0]-50,info["pos"][1]+50),
                 (info["pos"][0]+50,info["pos"][1]+50)]
            pygame.draw.polygon(screen,col,pts)
            screen.blit(small_font.render("Triangle",True,BLACK),
                        small_font.render("Triangle",True,BLACK).get_rect(center=(info["pos"][0],info["pos"][1]+20)))
        else:
            pygame.draw.rect(screen,col,(info["pos"][0]-70,info["pos"][1]-20,140,40))
            screen.blit(small_font.render("Do Nothing",True,BLACK),
                        small_font.render("Do Nothing",True,BLACK).get_rect(center=info["pos"]))

def draw_static_dot():            # ---  always‑visible dot (MOD‑4)  --------
    pygame.draw.circle(screen,YELLOW,(350,350),100)
    screen.blit(small_font.render("Focus Here",True,BLACK),
                small_font.render("Focus Here",True,BLACK).get_rect(center=(350,350)))

def draw_canvas():
    pygame.draw.rect(screen,WHITE,(canvas_x,canvas_y,canvas_w,canvas_h))
    pygame.draw.rect(screen,BLACK,(canvas_x,canvas_y,canvas_w,canvas_h),2)

def place_shape(shape,is_large=False,y_off=0):
    if shape=="Do Nothing": return
    cx,cy=canvas_x+250,canvas_y+200+y_off
    col=flicker_info[shape]["color"]
    if shape=="Circle":
        pygame.draw.circle(screen,col,(cx,cy),70 if is_large else 50)
    elif shape=="Rectangle":
        w,h=(140,80) if is_large else (100,60)
        pygame.draw.rect(screen,col,(cx-w/2,cy-h/2,w,h))
    elif shape=="Triangle":
        off=70 if is_large else 50
        pts=[(cx,cy-(90 if is_large else 70)),(cx-off,cy+off/2),(cx+off,cy+off/2)]
        pygame.draw.polygon(screen,col,pts)

def draw_focus_meter(v):
    mx,my,mw,mh=1050,150,40,400
    pygame.draw.rect(screen,WHITE,(mx,my,mw,mh),2)
    fh=int(mh*v/100)
    pygame.draw.rect(screen,interpolate(v),(mx,my+mh-fh,mw,fh))

# First, modify the session_plan to use a dynamic message:
session_plan = [
    ("Session Start",2, "Start"),
    ("Look at the flickering shape", 5, "Selection_Phase"),
    ("Shape Selected", 3, "Shape Selected"),
    ("Break - 5 sec", 1, "Break"),
    ("Focus/Unfocus - Look at the Dot", 5, "Static_Dot"),
    ("Size selected", 3, "Size selected"),
    # Changed to dynamic message - will be set during runtime
    ("", 2, "Shape Placement"),  
    ("Session END", 1, "END")
]

phase_markers={"Start":("1","2"),"Selection_Phase":("3","4"),
 "Shape Selected":("5","6"),"Break":("7","8"),"Static_Dot":("9","10"),
 "Size selected":("11","12"),"Shape Placement":("13","14"),"END":("15","16")}


def main():
    global selected_shape, last_placed_shape, last_placed_size
    placed_shapes, placed_large = [], []
    clock = pygame.time.Clock()

    for caption, dur, phase in session_plan:
        t0 = time.time()
        if phase in phase_markers: 
            event_marker(phase_markers[phase][0])
        
        current_caption = caption
        
        if phase == "Selection_Phase": 
            selected_shape = last_placed_shape = last_placed_size = None
        
        running = True
        while running:
            if time.time() - t0 >= dur: 
                running = False
            
            # Event handling (unchanged from working version)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: 
                    pygame.quit()
                    sock.close()
                    return
                
                if ev.type == pygame.KEYDOWN:
                    # Selection in Shape Selected phase
                    if phase == "Shape Selected" and ev.key in key_to_shape_map:
                        selected_shape = key_to_shape_map[ev.key]
                        event_marker(f"Shape_Selected_{selected_shape}")
                        print(f"Selected: {selected_shape}")
                    
                    # Action handling in Size selected phase
                    elif phase == "Size selected" and selected_shape:
                        if selected_shape == "Do Nothing" and ev.key in SPACE_KEYS|ENTER_KEYS:
                            last_placed_shape, last_placed_size = "Do Nothing", "N/A"
                            event_marker("Confirmed_Do_Nothing")
                            print("Confirmed: Do Nothing")
                            running = False
                        elif selected_shape != "Do Nothing":
                            if ev.key in SPACE_KEYS:
                                last_placed_shape, last_placed_size = selected_shape, "Small"
                                event_marker(f"Placed_Small_{selected_shape}")
                                print(f"Placed Small: {selected_shape}")
                                running = False
                            elif ev.key in ENTER_KEYS:
                                last_placed_shape, last_placed_size = selected_shape, "Large"
                                event_marker(f"Placed_Large_{selected_shape}")
                                print(f"Placed Large: {selected_shape}")
                                running = False
            
            # Drawing (unchanged)
            screen.fill(BLACK)
            draw_canvas()
            draw_flicker(phase)
            draw_static_dot()
            
            if phase == "Shape Placement":
                if last_placed_shape == "Do Nothing":
                    current_caption = "Nothing Selected"
                    nothing_text = font.render("Nothing Selected", True, WHITE)
                    text_rect = nothing_text.get_rect(
                        center=(canvas_x + canvas_w//2, canvas_y + canvas_h//2)
                    )
                    screen.blit(nothing_text, text_rect)
                elif last_placed_shape:
                    current_caption = f"Shape Placed: {last_placed_shape} ({last_placed_size})"
                    place_shape(last_placed_shape, last_placed_size == "Large")
            
            # Draw placed shapes
            for i, sh in enumerate(placed_shapes):
                place_shape(sh, placed_large[i], y_off=i*140)
            
            # Focus meter
            focus_val = (pygame.time.get_ticks()//50)%200
            focus_val = 100 - abs(focus_val - 100)
            draw_focus_meter(focus_val)
            
            # Display caption
            screen.blit(font.render(current_caption, True, WHITE), 
                       (canvas_x, canvas_y - 40))
            pygame.display.update()
            clock.tick(60)

        if phase in phase_markers: 
            event_marker(phase_markers[phase][1])
        
        if phase == "Size selected" and last_placed_shape and last_placed_shape != "Do Nothing":
            placed_shapes.append(last_placed_shape)
            placed_large.append(last_placed_size == "Large")

    # Session complete
    screen.fill(BLACK)
    screen.blit(font.render("Session Complete. Thank you!", True, WHITE),
               font.render("Session Complete. Thank you!", True, WHITE)
               .get_rect(center=(screen_width//2, screen_height//2)))
    pygame.display.update()
    pygame.time.delay(3000)
    pygame.quit()
    sock.close()

if __name__ == "__main__":
    main()

