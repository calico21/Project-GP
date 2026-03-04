import socket
import struct
import time
import math

# --- CONFIGURATION ---
SERVER_IP = '127.0.0.1'
PORT_SEND = 5000     # Send to the server's receive port
PORT_RECEIVE = 5001  # Listen on the server's broadcast port

def main():
    print("="*50)
    print(" Project-GP: Dummy UDP Client ")
    print("="*50)
    
    # 1. Setup Sockets
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind the receive socket to listen for the server's output
    sock_recv.bind(('0.0.0.0', PORT_RECEIVE))
    sock_recv.settimeout(0.1) # Non-blocking with a slight timeout

    # Struct formats matching the server
    tx_fmt = '<3f'   # steer, throttle_force, brake_force
    rx_fmt = '<11f'  # x, y, z, roll, pitch, yaw, z_fl, z_fr, z_rl, z_rr, mz

    print("Sending synthetic inputs... Press Ctrl+C to stop.")
    
    start_time = time.time()
    
    try:
        while True:
            t = time.time() - start_time
            
            # --- A. Generate Synthetic Driver Inputs ---
            # Gentle sine wave steering (approx +/- 0.2 radians)
            steer = math.sin(t * 1.5) * 0.2 
            
            # Pulsing throttle between 0% and 50%, converted to NEWTONS
            throttle_pct = (math.sin(t * 0.5) + 1.0) * 0.25 
            throttle_force = throttle_pct * 2000.0  # Max force of ~1000 N
            brake_force = 0.0

            # --- B. Send to Physics Server ---
            tx_data = struct.pack(tx_fmt, steer, throttle_force, brake_force)
            sock_send.sendto(tx_data, (SERVER_IP, PORT_SEND))

            # --- C. Receive and Read Vehicle State ---
            try:
                data, _ = sock_recv.recvfrom(44) # 11 floats * 4 bytes
                unpacked = struct.unpack(rx_fmt, data)
                
                x, y, z = unpacked[0:3]
                roll, pitch, yaw = unpacked[3:6]
                z_fl, z_fr, z_rl, z_rr = unpacked[6:10]
                mz = unpacked[10]
                
                # --- D. Dashboard Output ---
                print(f"Time: {t:05.2f}s | In: [St: {steer:+.2f}, Th(N): {throttle_force:4.0f}] | "
                      f"Out: Pos(x:{x:+.2f}, y:{y:+.2f}) Yaw:{yaw:+.2f}")
            
            except socket.timeout:
                print("Waiting for JAX physics server...")

            # Run the client at roughly 60Hz (Server runs physics internally at 200Hz)
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\n[System] Dummy client safely terminated.")
    finally:
        sock_send.close()
        sock_recv.close()

if __name__ == "__main__":
    main()