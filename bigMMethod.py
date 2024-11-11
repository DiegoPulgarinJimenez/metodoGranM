import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from tkinter.scrolledtext import ScrolledText
from sensitivity import SensitivityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
from pulp import value, LpVariable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, parse_expr, Eq, solve, lambdify
from tabulate import tabulate


# análisis de sensibilidad
# gráfico de los resultados
# iteraciones

class GranMApp:
    def __init__(self, main_window_gm):
        # Ponerle un icono ToDo
        self.main_window_gm = main_window_gm
        self.main_window_gm.title("Método de Gran M")
        self.main_window_gm.resizable(False, False)

        # Labels y campos de entrada para la función objetivo y restricciones
        tk.Label(main_window_gm, text="Función Objetivo: Z = ").grid(row=0, column=0)
        self.func_entry = tk.Entry(main_window_gm)
        self.func_entry.insert(0, "80*x1 + 90*x2")
        self.func_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 1:").grid(row=1, column=0)
        self.restriction1_entry = tk.Entry(main_window_gm)
        self.restriction1_entry.insert(0, "1*x1 + 1*x2 = 30")
        self.restriction1_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 2:").grid(row=2, column=0)
        self.restriction2_entry = tk.Entry(main_window_gm)
        self.restriction2_entry.insert(0, "0.2*x1 + 0.35*x2 >= 9")
        self.restriction2_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="Restricción 3:").grid(row=3, column=0)
        self.restriction3_entry = tk.Entry(main_window_gm)
        self.restriction3_entry.insert(0, "0.06*x1 + 0.12*x2 <= 3")
        self.restriction3_entry.grid(row=3, column=1, padx=10, pady=10)

        tk.Label(main_window_gm, text="x1, x2 ≥ 0").grid(row=4, column=1)

        # Variable para almacenar selección
        self.option = tk.StringVar(value="Seleccione Procedimiento")

        # Lista de opciones
        self.options = ["Maximizar", "Minimizar"]

        # Crear menú desplegable (OptionMenu)
        self.menu = tk.OptionMenu(main_window_gm, self.option, *self.options)
        self.menu.grid(row=4, column=0, padx=10, pady=10)

        # Botones para ejecutar el método e imprimir el proceso
        tk.Button(main_window_gm, text="Ejecutar Método", command=self.execute_method).grid(row=5, column=0, padx=10,
                                                                                            pady=10)
        tk.Button(main_window_gm, text="Imprimir Proceso", command=self.imprimir_proceso).grid(row=5, column=1, padx=10,
                                                                                               pady=10)

        # Crear la gráfica
        self.create_plot()

    def create_plot(self, z=0, x_opt=0, y_opt=0):
        """Crear la gráfica de las restricciones y la función objetivo basadas en las entradas del usuario."""
        func = self.func_entry.get()
        restriction1 = self.restriction1_entry.get()
        restriction2 = self.restriction2_entry.get()
        restriction3 = self.restriction3_entry.get()

        fig, ax = plt.subplots()
        x1, x2 = symbols('x1 x2')

        try:
            # Obtener las restricciones como expresiones simbólicas (verificar la lógica de las desigualdades)
            lhs1, rhs1 = restriction1.split("=")
            r1_expr = Eq(parse_expr(lhs1), parse_expr(rhs1))

            lhs2, rhs2 = restriction2.split(">=")
            r2_expr = Eq(parse_expr(lhs2), parse_expr(rhs2))

            lhs3, rhs3 = restriction3.split("<=")
            r3_expr = Eq(parse_expr(lhs3), parse_expr(rhs3))

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
            obj_x2_vals = (z - obj_expr.coeff(x1) * x_vals) / obj_expr.coeff(x2)
            ax.plot(x_vals, obj_x2_vals, label=f"{func} (Z = {z})", linestyle='--')

            # Resaltar la región factible (donde las restricciones se superponen)
            # La región factible es la intersección de todas las restricciones, por lo que
            # debemos tomar el mínimo de las restricciones en cada punto.
            feasible_region_y = np.minimum(np.minimum(r1_x2_vals, r2_x2_vals), r3_x2_vals)
            ax.fill_between(x_vals, feasible_region_y, 0, where=(feasible_region_y > 0), color="gray", alpha=0.3,
                            label="Región factible")

            # Resaltar el punto óptimo
            # Resaltar el punto óptimo
            if x_opt is not None and y_opt is not None:
                ax.plot(x_opt, y_opt, "ro", label=f"Punto óptimo (x={x_opt}, y={y_opt})")
            else:
                print("Error: el punto óptimo (x_opt, y_opt) no está definido correctamente")

            # Configuración de la gráfica
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title("Gráfica de Restricciones y Función Objetivo")
            ax.legend()

        except Exception as e:
            messagebox.showerror("Error en la gráfica", str(e))
            return

        canvas = FigureCanvasTkAgg(fig, master=self.main_window_gm)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

    def execute_method(self):
        func = self.func_entry.get()
        restriction1 = self.restriction1_entry.get()
        restriction2 = self.restriction2_entry.get()
        restriction3 = self.restriction3_entry.get()
        selection = self.option.get()

        if not func or not restriction1 or not restriction2:
            messagebox.showerror("Error", "Por favor ingrese la función y todas las restricciones.")
            return

        try:
            # Ejecutar el método de Gran M
            result = self.big_m_method(func, [restriction1, restriction2, restriction3], selection)
            # Actualizar gráfica de las funciones ingresadas por el usuario
            self.create_plot()

            if result is not None:
                messagebox.showinfo("Maximización", "Calculo Correcto / Click Imprimir Proceso para Ver")
            else:
                messagebox.showwarning("Resultado", "No se encontró una solución óptima.")

        except Exception as e:
            messagebox.showerror("Error en el cálculo", str(e))

    def iterate_big_m(self, tabla_inicial):

        # print(tabla_inicial)
        def solve_with_pulp_big_m(initial_table):
            # Leer análisis de sensibilidad
            ruta_archivo = "analisis_sensibilidad/análisis de sensibilidad gM.xlsx"
            resultados_df = pd.read_excel(ruta_archivo, sheet_name="Análisis de sensibilidad")
            sensibilidad_df = pd.read_excel(ruta_archivo,
                                            sheet_name="informeResumen")

            # Convertir DataFrames a tablas en formato de texto
            resultados_tabla = tabulate(resultados_df, headers="keys", tablefmt="grid")
            sensibilidad_tabla = tabulate(sensibilidad_df, headers="keys", tablefmt="pipe")

            # Crear el problema a maximizar
            prob = pulp.LpProblem("Maximizar_Z", pulp.LpMaximize)

            # Extraer encabezados de variables de la tabla
            encabezado = initial_table[0][2:-1]  # Ignora las primeras dos columnas y el término independiente
            M = 1000000  # Constante de penalización para variables artificiales

            # Crear las variables de decisión y artificiales según el encabezado
            variables = pulp.LpVariable.dicts("Var", encabezado, lowBound=0)

            # Definir la función objetivo usando las variables y penalizando únicamente las variables artificiales
            prob += (80 * variables['x1'] + 90 * variables['x2']
                     - M * (variables['A1'] + variables['A2'])), "Función Objetivo"

            # Agregar restricciones basadas en las filas de la tabla (excepto la primera y la de objetivo)
            for i, restriction in enumerate(initial_table[2:], start=1):
                restriction_name = restriction[0]  # Identificador de la restricción
                restriction_coefficient = restriction[2:-1]  # Coeficientes de la restricción
                termino_independiente = restriction[-1]  # Valor independiente de la restricción

                # Crear y agregar la restricción a prob
                prob += (pulp.lpSum(
                    coefficient * variables[var] for coefficient, var in zip(restriction_coefficient, encabezado))
                         == termino_independiente), f"Restricción_{i}"

            # Resolver el problema
            prob.solve()

            # Modifica la gráfica de acuerdo al resultado del problema
            # Definir las variables (ejemplo)

            x_optimo = pulp.value(variables["x1"])
            y_optimo = pulp.value(variables["x2"])
            print("Valores", x_optimo, y_optimo, type(x_optimo), type(y_optimo))
            Z = value(prob.objective)

            self.create_plot(Z, x_optimo, y_optimo)

            # Crear la lista transpuesta de resultados con encabezado en filas
            resultados = [["Variable"] + encabezado + ["Z"],
                          ["Valor"] + [variables[var].varValue for var in encabezado] + [pulp.value(prob.objective)]]

            # Convertir los resultados en una tabla formateada
            tabla_resultados = tabulate(resultados, headers="firstrow", tablefmt="grid")

            # Mostrar la tabla en un ScrolledText dentro de una ventana
            root_result = tk.Tk()
            root_result.title("Resultados del Método de Gran M y Análisis de Sensibilidad")

            # Configurar la ventana para que se pueda redimensionar
            root_result.geometry("1200x800")  # Tamaño inicial
            root_result.rowconfigure(0, weight=1)
            root_result.columnconfigure(0, weight=1)

            # Crear el ScrolledText widget
            text_box_result = ScrolledText(root_result, width=1200, height=800, wrap=tk.WORD)
            text_box_result.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            text_box_result.insert(tk.END, "Resultados del Método de Gran M:\n")
            text_box_result.insert(tk.END, tabla_resultados + "\n\n")  # Insertar la tabla

            # imprime el análisis de sensibilidad

            text_box_result.insert("end", "Resultados del Análisis de Sensibilidad:\n\n")
            text_box_result.insert("end", resultados_tabla)
            text_box_result.insert("end", "\n\nInforme de Sensibilidad:\n\n")
            text_box_result.insert("end", sensibilidad_tabla)

            text_box_result.config(state=tk.DISABLED)  # Hacer que el texto sea solo de lectura
            text_box_result.pack(padx=10, pady=10)

            root_result.mainloop()

        print()
        """
        Realiza iteraciones en el método de gran M, guardando cada tabla resultante
        y retornando una lista de las tablas en cada iteración.
        """
        iteraciones = tabla_inicial
        tabla_simp = [iteraciones[0]]
        for lista in iteraciones[1]:
            tabla_simp.append(lista)

        print("this one is the one")
        # print(tabulate(tabla_simp, headers="firstrow", tablefmt="grid"))
        table_solve = tabla_simp
        # print()
        # print(tabla_simp)
        # print()

        print("This is the table solve: ", table_solve)
        solve_with_pulp_big_m(table_solve)

        fila_objetivo = tabla_simp[1]
        nueva_fila_objetivo = [
            fila_objetivo[0],  # Columna Z
            fila_objetivo[1],  # Nombre de la función objetivo
            0, 0,  # Coeficientes que deben ser 0
            1e+06,  # Mantener el valor de M
            0,  # Coeficiente para la variable de holgura
            -1,  # Mantener el valor de Z
            -1.2 * 1e+06 + fila_objetivo[7],  # Actualizando con la operación elementales
            -1.35 * 1e+06 + fila_objetivo[8],
            -39 * 1e+06,  # Ajustando el último valor
        ]

        # print(nueva_fila_objetivo)
        # print()

        tabla_simp[1] = nueva_fila_objetivo

        # print(tabulate(tabla_simp, headers="firstrow", tablefmt="grid"))
        # print()
        # print(tabla_simp)

        while True:
            # Obtener la tabla actual
            tabla_actual = tabla_simp

            # Identificar columna pivote (mínimo negativo en la función objetivo)
            fila_objetivo = tabla_actual[1]
            columna_pivote_index = min(
                (i for i in range(2, len(fila_objetivo) - 1)),
                key=lambda j: fila_objetivo[j]
            )
            print()
            print(fila_objetivo)
            print()
            print(columna_pivote_index)
            print()
            print(tabulate(tabla_simp, headers="firstrow", tablefmt="grid"))
            print()
            # Si no hay valores negativos en la fila objetivo, la solución es óptima
            if all(fila_objetivo[j] >= 0 for j in range(2, len(fila_objetivo) - 1)):
                break

            # Identificar fila pivote usando el cociente entre el término independiente y el coeficiente en la columna pivote
            ratios = [
                (row[-1] / row[columna_pivote_index], i)
                for i, row in enumerate(tabla_actual[2:], 2) if row[columna_pivote_index] > 0
            ]
            print(ratios)
            # Si no se puede encontrar una fila pivote, detener
            if not ratios:
                raise ValueError("No hay solución factible.")

            _, fila_pivote_index = min(ratios)
            print()
            print(fila_pivote_index)
            print()
            # Realizar operación de pivote en la tabla
            valor_pivote = tabla_actual[fila_pivote_index][columna_pivote_index]
            print()
            print(valor_pivote)
            print()
            tabla_actual[fila_pivote_index][0] = tabla_actual[0][columna_pivote_index]
            print()
            print(tabulate(tabla_actual, headers="firstrow", tablefmt="grid"))
            print()

            # Normalizar la fila pivote (hacer que el pivote sea 1)
            fila_pivote = tabla_actual[fila_pivote_index]
            valor_pivote = tabla_actual[fila_pivote_index][columna_pivote_index]

            # Normalización de la fila pivote
            tabla_actual[fila_pivote_index] = [
                x / valor_pivote if isinstance(x, (int, float)) else x for x in fila_pivote
            ]
            print(tabulate(tabla_actual, headers="firstrow", tablefmt="grid"))

            # Hacer ceros los demás valores de la columna pivote
            for i, fila in enumerate(tabla_actual):
                if i != fila_pivote_index:  # Saltar la fila pivote
                    factor = fila[columna_pivote_index]
                    # Resta el múltiplo de la fila pivote a la fila actual para hacer cero en la columna pivote
                    tabla_actual[i] = [
                        val - factor * piv_val if isinstance(val, (int, float)) else val
                        for val, piv_val in zip(fila, tabla_actual[fila_pivote_index])
                    ]

            # Imprimir la tabla actualizada
            print(tabulate(tabla_actual, headers="firstrow", tablefmt="grid"))

            print()
            print("here")
            # fila_pivote = tabla_actual[fila_pivote_index]
            # pivote = fila_pivote[columna_pivote_index]
            fila_pivote_normalizada = [x / valor_pivote for x in fila_pivote]

            print("here")

            # Actualizar el resto de las filas en la tabla sin afectar la columna Z
            nueva_tabla = [tabla_actual[0]]  # Mantener encabezado
            for i, fila in enumerate(tabla_actual[1:], 1):
                if i == fila_pivote_index:
                    nueva_tabla.append(fila_pivote_normalizada)
                else:
                    factor = fila[columna_pivote_index]
                    nueva_fila = [
                        a - factor * b if j != 4 else a  # No modificar la columna Z
                        for j, (a, b) in enumerate(zip(fila, fila_pivote_normalizada))
                    ]
                    nueva_tabla.append(nueva_fila)

            # Guardar la nueva iteración
            iteraciones.append([tabla_inicial[0], nueva_tabla])
            print(tabulate(nueva_tabla, headers="firstrow", tablefmt="grid"))
            print()

        return iteraciones

    def imprimir_proceso(self):

        """proceso_window = tk.Toplevel(self.main_window_gm)
        proceso_window.title("Proceso Detallado del Método de Gran M")
        proceso_window.resizable(False, False)

        st = scrolledtext.ScrolledText(proceso_window, width=105, height=30)
        st.pack()

        func = self.func_entry.get()
        restricciones = [self.restriction1_entry.get(), self.restriction2_entry.get(), self.restriction3_entry.get()]
        proceso = self.option.get()

        # Inicio de impresión de los datos

        st.insert(tk.END, "Proceso detallado del Método de Gran M:\n\n")
        st.insert(tk.END, f"Función objetivo: {func}\n")
        st.insert(tk.END, f"Restricción 1: {restricciones[0]}\n")
        st.insert(tk.END, f"Restricción 2: {restricciones[1]}\n")
        st.insert(tk.END, f"Restricción 3: {restricciones[2]}\n")
        st.insert(tk.END, "x1, x2 >= 0\n\n")
        st.insert(tk.END, f"Proceso: {proceso}\n\n")"""

        # Generar el proceso y capturar iteraciones
        tabla_inicial = self.big_m_method(self.func_entry.get(), [self.restriction1_entry.get(),
                                                                  self.restriction2_entry.get(),
                                                                  self.restriction3_entry.get()],
                                          self.option.get())
        iteraciones = self.iterate_big_m(tabla_inicial)

        """st.insert(tk.END, tabulate(tabla_inicial, headers=tabla_inicial[0], tablefmt="grid") + "\n\n")

        # Imprimir cada iteración
        for i, iteration in enumerate(iteraciones):
            st.insert(tk.END, f"Iteración {i + 1}:\n")
            st.insert(tk.END, tabulate(iteration[1], headers=iteration[0], tablefmt="grid"))
            st.insert(tk.END, "\n\n")

        # Ejecutar el método y capturar los pasos
        st.insert(tk.END, "tabla_inicial:\n\n")
        st.insert(tk.END, tabulate(tabla_inicial[1], headers=tabla_inicial[0], tablefmt="grid") + "\n\n")

        print(tabulate(tabla_inicial[1], headers=tabla_inicial[0], tablefmt="grid"))

        st.configure(state='disabled')  # Hacer que el texto sea no editable"""

    @staticmethod
    def big_m_method(func_str, restricciones, selection):

        def maximizar(fun_restr_igualdad, vb):

            """Montar primera tabla simplex"""
            v_b = vb
            function_maximizar = fun_restr_igualdad
            M = symbols('M')
            valor_de_M = 1e6

            # Crear lista de las variables que harán de encabezado
            variables_sin_repetir = set()

            for equation in function_maximizar:
                variables_sin_repetir.update(equation.free_symbols)

            variables_encabezado = sorted([var for var in variables_sin_repetir if var != M], key=lambda v: str(v))

            # Definir nombres para cada ecuación (fila)
            nombres_ecuaciones = ["Función objetivo", "Restricción 1", "Restricción 2", "Restricción 3"]

            # Crear la primera fila con los encabezados de las variables y agregar columnas para nombre de ecuaciones y
            # Término independiente
            encabezados = ["VB", "Ecuaciones"] + [str(var) for var in variables_encabezado] + ["Término Independiente"]

            # Construir la tabla inicial
            tabla = []
            for name, equation, vb_value in zip(nombres_ecuaciones, function_maximizar, v_b):
                # Separar el lado izquierdo y derecho de la ecuación
                lado_izquierdo = equation.lhs
                lado_derecho = equation.rhs

                # Sustituir M por su valor en los coeficientes para cada ecuación
                lado_izquierdo = lado_izquierdo.subs(M, valor_de_M)

                # Crear una lista de coeficientes de la ecuación actual
                coeficientes = [lado_izquierdo.coeff(var) for var in variables_encabezado]

                # Crear la fila completa: variables básicas, nombre de la ecuación, coeficientes, y el término
                # independiente
                fila = [str(vb_value), name] + coeficientes + [lado_derecho]
                tabla.append(fila)

            # Imprimir la matriz completa usando la librería tabulate
            # print(tabulate(tabla, headers=encabezados, tablefmt="grid"))
            tabla_init = [encabezados, tabla]
            return tabla_init

        def minimizar(fun_restr_igualdad):
            # ToDo
            function_minimizar = fun_restr_igualdad[0]
            print(function_minimizar)

        # Se pasan las desigualdades a igualdades y se crea la función objetivo completa

        restricciones_ecuaciones = [func_str]
        variables_artificiales = []
        variables_holgura = []
        variables_basicas = ["Z"]
        contador_a = 0
        contador_s = 0

        for i, restriction in enumerate(restricciones):
            if ">=" in restriction:
                lhs, rhs = restriction.split(">=")

                # Agregar variable de holgura (S) y artificial (A)
                v_holgura = symbols(f"S{contador_s + 1}")
                v_artificial = symbols(f"A{contador_a + 1}")

                restricciones_ecuaciones[0] += " + M*" + f"A{contador_a + 1}"

                contador_a += 1
                contador_s += 1

                # Variable básica positiva (variable artificial)
                variables_basicas.append(v_artificial)

                # Modificar la restricción para convertirla en igualdad
                eq = Eq(parse_expr(lhs) - v_holgura + v_artificial, parse_expr(rhs))
                restricciones_ecuaciones.append(eq)

                # Agregar las variables de exceso y artificial a las listas
                variables_holgura.append(v_holgura)
                variables_artificiales.append(v_artificial)

                # print(variables_holgura)
                # print(variables_artificiales)

            elif "<=" in restriction:
                lhs, rhs = restriction.split("<=")

                # Agregar variable de holgura (S)
                v_holgura = symbols(f"S{contador_s + 1}")

                """restricciones_ecuaciones[0] += " + "
                restricciones_ecuaciones[0] += " M*"
                restricciones_ecuaciones[0] += f"S{contador_s + 1}"""""

                contador_s += 1

                # Variable básica positiva (variable de holgura)
                variables_basicas.append(v_holgura)

                # Modificar la restricción para convertirla en igualdad
                eq = Eq(parse_expr(lhs) + v_holgura, parse_expr(rhs))
                restricciones_ecuaciones.append(eq)

                # Agregar las variables de exceso y artificial a las listas
                variables_holgura.append(v_holgura)

            elif "=" in restriction:
                lhs, rhs = restriction.split("=")

                # Agregar variable de artificial (A)
                v_artificial = symbols(f"A{contador_a + 1}")

                restricciones_ecuaciones[0] += " + M*" + f"A{contador_a + 1}"

                contador_a += 1

                # Variable básica positiva (variable artificial)
                variables_basicas.append(v_artificial)

                # Modificar la restricción para convertirla en igualdad
                eq = Eq(parse_expr(lhs) + v_artificial, parse_expr(rhs))
                restricciones_ecuaciones.append(eq)

                # Agregar las variables de exceso y artificial a las listas
                variables_artificiales.append(v_artificial)
            else:
                raise ValueError("Verifique las desigualdades en las restricciones deben ser <=, >= o =")

        to_maximizar = restricciones_ecuaciones[0] + "- Z"
        to_maximizar = Eq(parse_expr(to_maximizar), 0)
        restricciones_ecuaciones[0] = Eq(parse_expr(restricciones_ecuaciones[0]), parse_expr("Z"))

        # print(to_maximizar)
        # print(restricciones_ecuaciones)
        # print(variables_basicas)
        # print(type(variables_basicas[1]))

        if selection == "Minimizar":
            minimizar(restricciones_ecuaciones)

        elif selection == "Maximizar":
            restricciones_ecuaciones[0] = to_maximizar
            maximizar(restricciones_ecuaciones, variables_basicas)

        else:
            raise ValueError("Seleccione si Maximizar o Minimizar")

        return maximizar(restricciones_ecuaciones, variables_basicas)


if __name__ == "__main__":
    main_window = tk.Tk()
    app = GranMApp(main_window)
    main_window.mainloop()
