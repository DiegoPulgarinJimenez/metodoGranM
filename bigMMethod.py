import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, parse_expr, Eq, solve, lambdify
import pandas as pd


class GranMApp:
    def __init__(self, main_window_gm):
        # Ponerle un icono ToDo
        self.main_window_gm = main_window_gm
        self.main_window_gm.title("Método de Gran M")
        self.main_window_gm.resizable(False, False)

        # Labels y campos de entrada para la función objetivo y restricciones
        tk.Label(main_window_gm, text="Función Objetivo: Z = ").grid(row=0, column=0)
        self.func_entry = tk.Entry(main_window_gm)
        self.func_entry.insert(0, "3*x1 + 5*x2")
        self.func_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 1:").grid(row=1, column=0)
        self.restriction1_entry = tk.Entry(main_window_gm)
        self.restriction1_entry.insert(0, "2*x1 + 3*x2 <= 12")
        self.restriction1_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 2:").grid(row=2, column=0)
        self.restriction2_entry = tk.Entry(main_window_gm)
        self.restriction2_entry.insert(0, "4*x1 + 1*x2 <= 5")
        self.restriction2_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 3:").grid(row=3, column=0)
        self.restriction3_entry = tk.Entry(main_window_gm)
        self.restriction3_entry.insert(0, "3*x1 + 3*x2 <= 18")
        self.restriction3_entry.grid(row=3, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="x1, x2 ≥ 0").grid(row=4, column=1)

        # Botones para ejecutar el método e imprimir el proceso
        tk.Button(main_window_gm, text="Ejecutar Método", command=self.execute_method).grid(row=5, column=0, padx=10,
                                                                                            pady=10)
        tk.Button(main_window_gm, text="Imprimir Proceso", command=self.imprimir_proceso).grid(row=5, column=1, padx=10,
                                                                                               pady=10)

        # Crear la gráfica
        self.create_plot()

    def create_plot(self):
        """Crear la gráfica de las restricciones y la función objetivo basadas en las entradas del usuario."""
        func = self.func_entry.get()
        restriction1 = self.restriction1_entry.get()
        restriction2 = self.restriction2_entry.get()
        restriction3 = self.restriction3_entry.get()

        fig, ax = plt.subplots()
        x1, x2 = symbols('x1 x2')

        try:
            # Obtener las restricciones como expresiones simbólicas
            r1_expr = parse_expr(restriction1.replace("<=", "-(") + ")")
            r2_expr = parse_expr(restriction2.replace("<=", "-(") + ")")
            r3_expr = parse_expr(restriction3.replace("<=", "-(") + ")")

            # Resolver las restricciones una vez
            r1_sol = solve(r1_expr, x2)[0]
            r2_sol = solve(r2_expr, x2)[0]
            r3_sol = solve(r3_expr, x2)[0]

            # Convertir las soluciones simbólicas en funciones numéricas
            r1_func = lambdify(x1, r1_sol, 'numpy')
            r2_func = lambdify(x1, r2_sol, 'numpy')
            r3_func = lambdify(x1, r3_sol, 'numpy')

            # Obtener la función objetivo
            obj_expr = parse_expr(func)

            # Crear un rango de valores para x1
            x_vals = np.linspace(0, 600, 400)

            # Evaluar las restricciones usando las funciones numéricas
            r1_x2_vals = r1_func(x_vals)
            r2_x2_vals = r2_func(x_vals)
            r3_x2_vals = r3_func(x_vals)

            # Graficar las restricciones
            ax.plot(x_vals, r1_x2_vals, label=f"{restriction1}")
            ax.plot(x_vals, r2_x2_vals, label=f"{restriction2}")
            ax.plot(x_vals, r3_x2_vals, label=f"{restriction3}")

            # Graficar la función objetivo para diferentes valores de Z
            obj_func = lambdify(x1, obj_expr, 'numpy')
            for z in range(5, 20, 5):
                obj_x2_vals = (z - obj_expr.coeff(x1) * x_vals) / obj_expr.coeff(x2)
                ax.plot(x_vals, obj_x2_vals, label=f"{func} (Z = {z})", linestyle='--')

        except Exception as e:
            messagebox.showerror("Error en la gráfica", str(e))
            return

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Gráfica de Restricciones y Función Objetivo")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.main_window_gm)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

    def execute_method(self):
        func = self.func_entry.get()
        restriction1 = self.restriction1_entry.get()
        restriction2 = self.restriction2_entry.get()
        restriction3 = self.restriction3_entry.get()

        if not func or not restriction1 or not restriction2:
            messagebox.showerror("Error", "Por favor ingrese la función y todas las restricciones.")
            return

        try:
            # Ejecutar el método de Gran M
            result = self.big_m_method(func, [restriction1, restriction2, restriction3])

            # Actualizar gráfica de las funciones ingresadas por el usuario
            self.create_plot()

            if result is not None and result[0] is not None:
                # `result[0]` contendrá la tabla final (DataFrame), y `result[1]` contendrá las iteraciones
                result_str = result[0].to_string(index=True)
                messagebox.showinfo("Resultado", f"Resultado óptimo:\n\n{result_str}")
            else:
                messagebox.showwarning("Resultado", "No se encontró una solución óptima.")
        except Exception as e:
            messagebox.showerror("Error en el cálculo", str(e))

    def imprimir_proceso(self):
        proceso_window = tk.Toplevel(self.main_window_gm)
        proceso_window.title("Proceso Detallado del Método de Gran M")
        proceso_window.resizable(False, False)

        st = scrolledtext.ScrolledText(proceso_window, width=100, height=30)
        st.pack()

        func = self.func_entry.get()
        restricciones = [self.restriction1_entry.get(), self.restriction2_entry.get(), self.restriction3_entry.get()]

        # Hacer texto no editable después de imprimir el proceso ToDo
        st.insert(tk.END, "Proceso detallado del Método de Gran M:\n\n")
        st.insert(tk.END, f"Función objetivo: {func}\n")
        st.insert(tk.END, f"Restricción 1: {restricciones[0]}\n")
        st.insert(tk.END, f"Restricción 2: {restricciones[1]}\n")
        st.insert(tk.END, f"Restricción 3: {restricciones[2]}\n")
        st.insert(tk.END, "x1, x2 >= 0\n\n")

        # Ejecutar el método y capturar los pasos
        tabla_inicial, tablas_iteraciones = self.big_m_method(func, restricciones)

        # Imprimir tabla inicial (Primera Iteración (Primer Sprint octubre 10 ToDo))
        st.insert(tk.END, "Tabla Inicial:\n")
        st.insert(tk.END, tabla_inicial.to_string(index=False))
        st.insert(tk.END, "\n\n")

        # Imprimir cada iteración (ToDo)
        for i, tabla in enumerate(tablas_iteraciones, 1):
            st.insert(tk.END, f"Iteración {i}:\n")
            st.insert(tk.END, tabla.to_string(index=False))
            st.insert(tk.END, "\n\n")
            # Explicación de la tabla ToDo

    @staticmethod
    def big_m_method(func_str, restricciones):
        # Método gran M ToDo
        return None


if __name__ == "__main__":
    main_window = tk.Tk()
    app = GranMApp(main_window)
    main_window.mainloop()
