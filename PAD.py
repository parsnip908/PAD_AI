
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygame



def forward_transform(input_array):
    """
    Converts a (5,6) integer array to a (7,5,6) one-hot reshaped array.
    
    Args:
        input_array (numpy.ndarray): A (5,6) array with values in range [0,6].
    
    Returns:
        numpy.ndarray: A (7,5,6) reshaped one-hot encoded array.
    """
    if input_array.shape != (5, 6):
        raise ValueError("Input array must have shape (5,6)")
    
    if np.any(input_array < 0) or np.any(input_array > 6):
        raise ValueError("All values in input_array must be in range [0,6]")

    # Step 1: Convert to one-hot encoding (5,6,7)
    expanded_array = np.eye(7)[input_array]

    # Step 2: Move the last axis (7) to the first position (7,5,6)
    reshaped_array = np.moveaxis(expanded_array, 2, 0)

    return reshaped_array


def reverse_transform(reshaped_array):
    """
    Converts a (7,5,6) one-hot reshaped array back to a (5,6) integer array.
    
    Args:
        reshaped_array (numpy.ndarray): A (7,5,6) one-hot encoded array.
    
    Returns:
        numpy.ndarray: A (5,6) integer array.
    """
    if reshaped_array.shape != (7, 5, 6):
        raise ValueError("Input array must have shape (7,5,6)")

    # Step 1: Move axis back to restore (5,6,7)
    restored_5_6_7 = np.moveaxis(reshaped_array, 0, 2)

    # Step 2: Use argmax() to recover original integer values
    recovered_input_array = np.argmax(restored_5_6_7, axis=2)

    return recovered_input_array



def visualize_array(input_array):
    """
    Visualizes a (5,6) integer array as a grid of colored circles.
    
    Args:
        input_array (numpy.ndarray): A (5,6) array with values in range [0,6].
    """
    if input_array.shape != (5, 6):
        raise ValueError("Input array must have shape (5,6)")

    # Define a colormap for values 0-6
    color_map = {
        0: "red", 1: "blue", 2: "green", 3: "yellow",
        4: "purple", 5: "pink", 6: "black"
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)

    # Draw circles at grid positions
    for i in range(5):
        for j in range(6):
            value = input_array[i, j]
            color = color_map[value]
            ax.add_patch(plt.Circle((j, 4 - i), 0.4, color=color, ec="black"))

            # Display the integer value inside the circle
            ax.text(j, 4 - i, str(value), ha='center', va='center', fontsize=12, color="white")

    # Formatting
    ax.set_xticks(range(6))
    ax.set_yticks(range(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_frame_on(False)

    # Show plot
    plt.show()

def visualize_array_pygame(input_array):
    """
    Visualizes a (5,6) integer array as a grid of colored circles using Pygame.

    Args:
        input_array (numpy.ndarray): A (5,6) array with values in range [0,6].
    """
    if input_array.shape != (5, 6):
        raise ValueError("Input array must have shape (5,6)")

    # Initialize Pygame
    pygame.init()
    
    # Set up display
    width, height = 600, 500  # Window size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Array Visualization")

    # Define colors for each integer value
    color_map = {
        0: (255, 0, 0),     # Red
        1: (0, 0, 255),     # Blue
        2: (0, 255, 0),     # Green
        3: (255, 255, 0),   # Yellow
        4: (143, 55, 237),   # Purple
        5: (242, 143, 209),   # Pink
        6: (0, 0, 0)    # Cyan
    }

    # Calculate circle positions
    circle_radius = 40
    padding = 10
    for i in range(5):
        for j in range(6):
            value = input_array[i, j]
            color = color_map[value]
            x = j * (circle_radius * 2 + padding) + circle_radius + padding
            y = i * (circle_radius * 2 + padding) + circle_radius + padding

            # Draw the circle
            if value == 5:
                pygame.draw.rect(screen, color, (x - circle_radius, y - circle_radius, circle_radius*2, circle_radius*2))
            else:
                pygame.draw.circle(screen, color, (x, y), circle_radius)
            # Draw the integer value
            # font = pygame.font.Font(None, 36)
            # text_surface = font.render(str(value), True, (255, 255, 255))
            # text_rect = text_surface.get_rect(center=(x, y))
            # screen.blit(text_surface, text_rect)

    # Update the display
    pygame.display.flip()

    # Event loop to keep the window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    # Example Usage
    # np.random.seed(42)  # For reproducibility

    # Generate input (5,6) array with values in [0,6]
    input_array = np.random.randint(0, 6, (5, 6))
    visualize_array_pygame(input_array)

    print("Original Input Array (5,6):\n", input_array)

    # Forward Transformation
    transformed_array = forward_transform(input_array)
    print("\nTransformed Array (7,5,6):\n", transformed_array.astype(int))  # Convert to int for readability

    # Reverse Transformation
    recovered_array = reverse_transform(transformed_array)
    print("\nRecovered Input Array (5,6):\n", recovered_array)

    # Validation
    assert np.array_equal(recovered_array, input_array), "Reverse operation failed!"
    print("\nSuccessfully recovered the original input array!")



