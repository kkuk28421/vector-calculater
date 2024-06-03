"""
code for caculating vectors and looking at the graph
if you have any help, please email us.

~~~~~~~~~~~~~~~~~~~~~~~

Copyright: (c) 2024 yeongjun hwang, jaeho kim
license: MIT license, see LICENSE for more details.
"""

import copy
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class VectorCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vector Calculator")
        self.resizable(False, False)
        self.vectors = []
        self.init_gui()


    def init_gui(self):
        self.frame_input = ttk.Frame(self, padding = "10")
        self.frame_input.grid(row = 0, column = 0, sticky = (tk.W, tk.E, tk.N, tk.S))

        self.frame_controls = ttk.Frame(self, padding = "10")
        self.frame_controls.grid(row = 1, column = 0, sticky = (tk.W, tk.E, tk.N, tk.S))

        self.frame_output = ttk.Frame(self, padding = "10")
        self.frame_output.grid(row = 2, column = 0, sticky = (tk.W, tk.E, tk.N, tk.S))

        self.frame_plot = ttk.Frame(self, padding = "10")
        self.frame_plot.grid(row = 0, column = 1, rowspan = 3, sticky = (tk.W, tk.E, tk.N, tk.S))

        ttk.Button(self.frame_controls, text = "Add Vector", command = self.add_vector).grid(column = 0, row = 0, sticky = tk.W)
        ttk.Button(self.frame_controls, text = "Sum", command = self.calculate_sum).grid(column = 2, row = 0, sticky = tk.W)
        ttk.Button(self.frame_controls, text = "Difference", command = self.calculate_difference).grid(column = 3, row = 0, sticky = tk.W)
        ttk.Button(self.frame_controls, text = "Inner product", command = self.calculate_inproduct).grid(column = 4, row = 0, sticky = tk.W)
        ttk.Button(self.frame_controls, text = "Outer product", command = self.calculate_outproduct).grid(column = 5, row = 0, sticky = tk.W)
        ttk.Button(self.frame_controls, text = "Reset", command = self.reset).grid(column = 0, row = 1, sticky = tk.W)
        
        ttk.Label(self.frame_controls, text = "").grid(column = 1, row = 1) # padding label

        self.scalar_entry = ttk.Entry(self.frame_controls, width = 5)
        self.scalar_entry.grid(column = 4, row = 1, sticky = tk.W)

        self.scalar_label = ttk.Label(self.frame_controls, text = "Scalar Value:")
        self.scalar_label.grid(column = 3, row = 1, sticky = tk.E)

        ttk.Button(self.frame_controls, text = "Multiplication", command = self.calculate_scalar).grid(column = 2, row = 1, sticky = tk.W)

        self.result_text = tk.Text(self.frame_output, width = 45, height = 10, font = ("Helvetica", 14))
        self.result_text.grid(row = 0, column = 0, sticky = (tk.W, tk.E, tk.N, tk.S))


    def add_vector(self):
        if len(self.vectors) == 5:
            messagebox.showerror("Error", "Maximum of 5 vectors allowed")
            return

        row = len(self.vectors) + 1
        entry_vector = ttk.Entry(self.frame_input, width = 20)
        entry_vector.grid(column = 1, row = row, sticky = tk.W)
        self.vectors.append(entry_vector)

        ttk.Label(self.frame_input, text = f"Vector {row} (comma separated):").grid(column = 0, row = row, sticky = tk.W)


    def get_vectors(self, t = 0):
        vectors = []

        for entry_vector in self.vectors:
            vec = np.array([float(x) for x in entry_vector.get().split(',')])
            vectors.append(vec)

        return vectors


    def calculate_sum(self):
        vectors = self.get_vectors()
        result_vector = np.sum(vectors, axis = 0)
        self.show_result(vectors, result_vector, 'Sum')
        self.plot_vectors(vectors, result_vector, 'Sum', 'g')


    def calculate_difference(self):
        vectors = self.get_vectors()
        result_vector = copy.deepcopy(vectors[0])

        for vec in vectors[1:]:
            result_vector -= vec

        self.show_result(vectors, result_vector, 'Difference')
        self.plot_vectors(vectors, result_vector, 'Difference', 'y')


    def calculate_scalar(self):
        scalar_value = self.scalar_entry.get()

        if scalar_value.strip():
            scalar = float(scalar_value)
            vectors = self.get_vectors()
            result_vector = scalar * vectors[0]
            self.show_result([vectors[0]], result_vector, 'Scalar Multiplication')
            self.plot_vectors([vectors[0]], result_vector, 'Scalar Multiplication', 'm')
        
        else:
            messagebox.showwarning("Warning", "Please enter a scalar value.")


    def show_result(self, vectors, result_vector, operation):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Operation: {operation}\n")

        for i, vec in enumerate(vectors):
            self.result_text.insert(tk.END, f"Vector {i+1}: {vec}\n")

        self.result_text.insert(tk.END, f"Result: \n{result_vector}\n")


    def calculate_inproduct(self):
        vectors = self.get_vectors()
        result_vector = np.dot(vectors[0], vectors[1])
        self.show_result(vectors, result_vector, 'dot product')

    
    def calculate_outproduct(self):        
        vectors = self.get_vectors()
        vectors0 = np.hstack((vectors[0], 0))
        vectors1 = np.hstack((vectors[1], 0))

        if len(vectors[0]) == 2 and len(vectors[1]) == 2:
            result_vector = np.cross(vectors0, vectors1)
            self.show_result(vectors, result_vector, 'outer product')
        else:
            result_vector = np.cross(vectors[0], vectors[1])
            self.show_result(vectors, result_vector, 'outer product')            


    def plot_vectors(self, vectors, result_vector, operation, color):
        for widget in self.frame_plot.winfo_children():
            widget.destroy()

        vector_a = copy.deepcopy(vectors)
        vector_a.append(result_vector)

        unit = self.calculate_xy(vectors)

        fig, ax = plt.subplots()
        origin = np.zeros_like(vectors[0])

        colors = ['r', 'b', 'c', 'm', 'g']
        start = origin[:2]

        for i, (vec, col) in enumerate(zip(vectors, colors)):
            if i == 0:
                ax.quiver(*start, *vec[:2], color = col, label = f'Vector {i+1} {vec}', scale = 1.0, scale_units = 'xy', angles = 'xy')

            else:
                end = start + vec[:2]
                ax.quiver(*start, *vec[:2], color = col, label = f'Vector {i+1} {vec}', scale = 1.0, scale_units = 'xy', angles = 'xy')
                start = end

        ax.set_xlim([-unit-2, unit+2])
        ax.set_ylim([-unit-2, unit+2])

        ax.axhline(0, color = 'grey', linewidth = 0.5)
        ax.axvline(0, color = 'grey', linewidth = 0.5)
        ax.grid(True)
        ax.legend()
        ax.set_title('Vector Operations')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        ax.set_aspect('equal', adjustable = 'box')

        canvas = FigureCanvasTkAgg(fig, master = self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill = tk.BOTH, expand = True)

        plt.switch_backend("TkAgg")


    def calculate_xy(self, vectors):
        x, y = 0, 0

        for vec in vectors:
            if abs(vec[0]) > x: x = abs(vec[0])
            if abs(vec[1]) > y: y = abs(vec[1])

        return x if x > y else y


    def reset(self):
        for widget in self.frame_input.winfo_children():
            widget.destroy()

        self.vectors = []
        self.result_text.delete(1.0, tk.END)

        for widget in self.frame_plot.winfo_children():
            widget.destroy()



if __name__ == "__main__":
    app = VectorCalculator()
    app.mainloop()