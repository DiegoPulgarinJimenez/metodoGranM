import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pulp import value
from sympy import symbols, parse_expr, Eq, solve, lambdify
from tabulate import tabulate


class GranMApp:
    def __init__(self, main_window_gm):

        # configuraciones de la interfaz
        icon_window = tk.PhotoImage(file="images_icon/big_m_icon.png")  # Carga el ícono
        self.main_window_gm = main_window_gm
        self.main_window_gm.title("Método de Gran M")
        self.main_window_gm.resizable(False, False)
        self.main_window_gm.iconphoto(True, icon_window)

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

            # Graficar la función objetivo así como el valor final de Z
            obj_x2_vals = (z - obj_expr.coeff(x1) * x_vals) / obj_expr.coeff(x2)
            ax.plot(x_vals, obj_x2_vals, label=f"{func} (Z = {z})", linestyle='--')

            # Resaltar la región factible (donde las restricciones se superponen)
            # La región factible es la intersección de todas las restricciones, por lo que
            # debemos tomar el mínimo de las restricciones en cada punto.
            feasible_region_y = np.minimum(np.minimum(r1_x2_vals, r2_x2_vals), r3_x2_vals)
            ax.fill_between(x_vals, feasible_region_y, 0, where=(feasible_region_y > 0), color="gray", alpha=0.3,
                            label="Región factible")

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
                messagebox.showinfo("Cálculo", "Cálculo Correcto / Click Imprimir Proceso Para Ver Detalles")
            else:
                messagebox.showwarning("Resultado", "No se encontró una solución óptima.")

        except Exception as e:
            messagebox.showerror("Error en el Cálculo", str(e))

    def iterate_big_m(self, tabla_inicial):

        """ tabla_simple correspondiente al parametro entrante en la función iterate_big_m, es igual al parametro
        entrante actual."""

        tabla_simple = [
            ['VB', 'Ecuaciones', 'A1', 'A2', 'S1', 'S2', 'Z', 'x1', 'x2', 'Término Independiente'],
            ['Z', 'Función objetivo', 1e6, 1e6, 0, 0, -1, 80, 90, 0],
            ['A1', 'Restricción 1', 1, 0, 0, 0, 0, 1, 1, 30],
            ['A2', 'Restricción 2', 0, 1, -1, 0, 0, 0.2, 0.35, 9],
            ['S2', 'Restricción 3', 0, 0, 0, 1, 0, 0.06, 0.12, 3]
        ]

        """print()
        tabla_inicial_2 = copy.deepcopy(tabla_inicial)
        iteraciones_tabla_ordenada = tabla_inicial_2[0:1] + tabla_inicial_2[1]
        print(iteraciones_tabla_ordenada)
        """

        def iteraciones_final(table_simplex):
            tabla_simplex_iteraciones = table_simplex

            # Reorganizar las columnas de la tabla
            def reorganizar_tabla(tabla, orden):
                # Crear índice del nuevo orden
                indices = [tabla[0].index(col) for col in orden]
                # Reordenar cada fila con base en los índices
                return [[fila[i] for i in indices] for fila in tabla]

            # Paso 2: Encontrar columna pivote
            def encontrar_columna_pivote(fila_objetivo_z, columnas_a_omitir, encabezados_1):
                indices_validos = [
                    i for i, col in enumerate(encabezados_1) if col not in columnas_a_omitir
                ]
                valores_validos = [fila_objetivo_z[i] for i in indices_validos]
                indice_pivote = indices_validos[valores_validos.index(min(valores_validos))]
                return indice_pivote

            # Paso 2: Encontrar fila pivote
            def encontrar_fila_pivote(tabla, index_columna_pivote):
                filas = tabla[2:]  # Ignorar encabezados y la fila Z
                cocientes = []
                for fila in filas:
                    termino_independiente = fila[-1]
                    value_columna_pivote = fila[index_columna_pivote]
                    if value_columna_pivote > 0:  # Evitar divisiones por 0 y cocientes negativos
                        cocientes.append(termino_independiente / value_columna_pivote)
                    else:
                        cocientes.append(float('inf'))  # Valores no válidos
                index_fila_pivote = cocientes.index(min(cocientes))
                return index_fila_pivote + 2  # Ajustar índice por encabezados y fila Z
                # return cocientes.index(min(cocientes)) + 2

            # Evaluar si aún hay valores negativos en la fila Z
            def tiene_valores_negativos(fila_objetivo_z, columnas_a_omitir, encabezados_1):
                indices_validos = [i for i, col in enumerate(encabezados_1) if col not in columnas_a_omitir]
                valores = [fila_objetivo_z[i] for i in indices_validos]
                return any(valor < 0 for valor in valores)

            # Copiar tabla para trabajar
            tabla_actualizada = [fila.copy() for fila in tabla_simplex_iteraciones]

            # Multiplicar las filas de A1 y A2 por -M (1e6)
            M = 1e6
            fila_A1 = tabla_actualizada[2]  # Fila A1
            fila_A2 = tabla_actualizada[3]  # Fila A2
            fila_Z = tabla_actualizada[1]  # Fila Z

            # Multiplicar A1 y A2 por -M
            fila_A1_negada = [-M * valor if isinstance(valor, (int, float)) else valor for valor in fila_A1]
            fila_A2_negada = [-M * valor if isinstance(valor, (int, float)) else valor for valor in fila_A2]

            # Sumar A1_negada, A2_negada y Z para calcular la nueva fila Z
            nueva_fila_Z = [
                (fila_Z[i] + fila_A1_negada[i] + fila_A2_negada[i]) if isinstance(fila_Z[i], (int, float)) else
                fila_Z[
                    i]
                for i in range(len(fila_Z))
            ]

            # Actualizar la fila Z en la tabla
            tabla_actualizada[1] = nueva_fila_Z

            # Orden deseado de las columnas
            orden_deseado = ['VB', 'Ecuaciones', 'Z', 'x1', 'x2', 'S1', 'S2', 'A1', 'A2', 'Término Independiente']

            # Reorganizar la tabla
            tabla_reorganizada = reorganizar_tabla(tabla_actualizada, orden_deseado)
            tabla_inicial_ordenada = reorganizar_tabla(tabla_simplex_iteraciones, orden_deseado)

            # Imprimir la tabla inicial ordenada
            print("\nTabla Inicial: ")
            print(tabulate(tabla_inicial_ordenada, headers='firstrow', tablefmt="fancy_grid"))

            # Imprimir la tabla reorganizada
            print()
            print("Tabla Simplex (actualización de fila Z, eliminando las M de A1 y A2):")
            print(tabulate(tabla_reorganizada, headers='firstrow', tablefmt='fancy_grid'))
            print()

            # Configuración
            columnas_ignoradas = ['VB', 'Ecuaciones', 'Z', 'Término Independiente']
            encabezados = tabla_reorganizada[0]
            fila_z = tabla_reorganizada[1]

            iteration = 1

            while tiene_valores_negativos(fila_z, columnas_ignoradas, encabezados):

                # Encontrar columna y fila pivote
                indice_columna_pivote = encontrar_columna_pivote(fila_z, columnas_ignoradas, encabezados)
                indice_fila_pivote = encontrar_fila_pivote(tabla_reorganizada, indice_columna_pivote)

                # Obtener número pivote
                numero_pivote = tabla_reorganizada[indice_fila_pivote][indice_columna_pivote]

                # Mostrar resultados
                columna_pivote = encabezados[indice_columna_pivote]
                fila_pivote = tabla_reorganizada[indice_fila_pivote]

                print(f"\nColumna pivote: {columna_pivote} (Índice: {indice_columna_pivote})")
                print(f"Fila pivote: {fila_pivote[0]} (Índice: {indice_fila_pivote})")
                print(f"Número pivote: {numero_pivote} \n")
                print(f"Tabla Iteración: {iteration}")

                # Tercer paso: Reemplazar la columna 'VB' de la fila pivote
                tabla_reorganizada[indice_fila_pivote][
                    0] = columna_pivote  # Reemplazar 'VB' por el nombre de la columna pivote

                # Hacer que el número pivote sea 1
                # Para eso, multiplicamos la fila pivote por el inverso del valor pivote
                tabla_reorganizada[indice_fila_pivote] = [
                    valor / numero_pivote if isinstance(valor, (int, float)) else valor
                    for valor in tabla_reorganizada[indice_fila_pivote]
                ]

                # Hacer ceros los valores de la columna pivote en las demás filas
                for i, fila in enumerate(tabla_reorganizada):
                    if i != indice_fila_pivote:  # Saltar la fila pivote
                        valor_columna_pivote = fila[indice_columna_pivote]
                        if isinstance(valor_columna_pivote, (int, float)) and valor_columna_pivote != 0:
                            factor = valor_columna_pivote
                            tabla_reorganizada[i] = [
                                fila[j] - (factor * tabla_reorganizada[indice_fila_pivote][j])
                                if isinstance(fila[j], (int, float)) else fila[j]
                                for j in range(len(fila))
                            ]

                print(tabulate(tabla_reorganizada, headers='firstrow', tablefmt='fancy_grid'))
                # Actualizar fila Z para verificar condición del `while`
                fila_z = tabla_reorganizada[1]
                iteration += 1

            # Imprimir la tabla final
            # Aplicar valor absoluto al valor de la Z, puesto que estamos maximizando
            fila_z_f = tabla_reorganizada[1]
            indice_valor_z = -1  # Última columna
            fila_z_f[indice_valor_z] = abs(fila_z_f[indice_valor_z])
            """print(fila_z_f[indice_valor_z])
            print(fila_z_f)"""
            print("\nTabla solución:")
            print(tabulate(tabla_reorganizada, headers='firstrow', tablefmt='fancy_grid'))

        # Llamamos a la función
        iteraciones_final(tabla_simple)

        def solve_with_pulp_big_m(initial_table):

            # Leer análisis de sensibilidad
            ruta_archivo = "analisis_sensibilidad/análisis de sensibilidad gM.xlsx"
            resultados_df = pd.read_excel(ruta_archivo, sheet_name="Análisis de sensibilidad")
            sensibilidad_df = pd.read_excel(ruta_archivo,
                                            sheet_name="informeResumen")

            # Convertir DataFrames a tablas en formato de texto
            resultados_tabla = tabulate(resultados_df, headers="keys", tablefmt="grid")
            sensibilidad_tabla = tabulate(sensibilidad_df, headers="keys", tablefmt="pipe")

            """Solución del modelo a través de la librería Pulp"""

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
                """restriction_name = restriction[0]  # Identificador de la restricción"""
                restriction_coefficient = restriction[2:-1]  # Coeficientes de la restricción
                termino_independiente = restriction[-1]  # Valor independiente de la restricción

                # Crear y agregar la restricción a prob
                prob += (pulp.lpSum(
                    coefficient * variables[var] for coefficient, var in zip(restriction_coefficient, encabezado))
                         == termino_independiente), f"Restricción_{i}"

            # Resolver el problema
            prob.solve()

            # Modifica la gráfica de acuerdo al resultado del problema
            x_optimo = pulp.value(variables["x1"])
            y_optimo = pulp.value(variables["x2"])
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

        # Preparación de la tabla (lista) para ser recibida como un parametro correcto
        iteraciones = tabla_inicial
        tabla_simp = [iteraciones[0]]
        for lista in iteraciones[1]:
            tabla_simp.append(lista)

        print()
        table_solve = tabla_simp

        solve_with_pulp_big_m(table_solve)

    def imprimir_proceso(self):

        # Generar el proceso y capturar iteraciones
        tabla_inicial = self.big_m_method(self.func_entry.get(), [self.restriction1_entry.get(),
                                                                  self.restriction2_entry.get(),
                                                                  self.restriction3_entry.get()],
                                          self.option.get())
        # iteraciones = self.iterate_big_m(tabla_inicial)
        self.iterate_big_m(tabla_inicial)

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

            # Imprimir la matriz completa usando la librería Tabulate
            tabla_init = [encabezados, tabla]
            return tabla_init

        def minimizar(fun_restr_igualdad, vb):
            """Invocamos maximizar puesto que los problemas de maximización y minimización son duales entre sí,
            si La región factible y la función objetivo están bien definidas y son consistentes el valor óptimo de la
            función objetivo en el problema primal (maximización) es igual al valor óptimo en el problema
            dual (minimización), siempre que ambos problemas tengan soluciones óptimas."""
            maximizar(fun_restr_igualdad, vb)

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
            restricciones_ecuaciones[0] = to_maximizar
            minimizar(restricciones_ecuaciones, variables_basicas)

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
