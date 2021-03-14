'''
Este script resuelve la ecuacion (1) del documento enunciado.pdf. La idea es
buscar el valor de a para el cual la integral toma un valor de 0.05.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import newton


def gausiana(x):
    """Funcion argumento de la integral de la ec. 1 del P1 de la tarea."""
    output = np.exp(-x**2/2) / np.sqrt(2*np.pi)
    return output


plt.figure(1, figsize=(6, 4))
plt.clf()

x_to_plot = np.linspace(-5, 5, 100)
plt.plot(x_to_plot, gausiana(x_to_plot))

plt.xlabel("$x$", fontsize=12)
plt.ylabel(r"$\exp(x^2/2)/\sqrt{2\pi}$", fontsize=12)
plt.subplots_adjust(top=0.95, right=0.95, left=0.15, bottom=0.15)
plt.show()

# Notese que esta funcion y el grafico no estan en el informe, sin embargo fue
# parte del proceso exploratorio para resolver el problema.


def gausiana_cv(u):
    """La funcion a integrar luego del cambio de variable u=1/y.
    """
    output = np.exp(-1/(2 * u**2)) / u**2 / np.sqrt(2*np.pi)
    return output


plt.figure(1, figsize=(6, 4))
plt.clf()

u_to_plot = np.linspace(1e-5, 10, 500)
plt.plot(u_to_plot, gausiana_cv(u_to_plot))

plt.xlabel("$u$", fontsize=12)
plt.ylabel(r"$\frac{1}{\sqrt{2\pi}u^2}\exp\left(\frac{-1}{2u^2}\right)$",
           fontsize=12)
plt.subplots_adjust(top=0.95, right=0.95, left=0.15, bottom=0.15)

plt.savefig("integral.pdf")
plt.show()


def integral_cv(a):
    """Calcula la integral de la funcion gaussiana_cv entre 0 y 1/a.
    """
    if type(a) == np.ndarray:
        output = [quad(gausiana_cv, 0, 1/a_i)[0] for a_i in a]
        output = np.array(output)
    else:
        output = quad(gausiana_cv, 0, 1/a)[0]
    return output


# A continuacion, una prueba de que la funcion integral_cv entrega resultados
# razonables
print("integral_cv(a=inf) = {:.5f}".format(integral_cv(1e4)))  # deberia dar ~0
print("integral_cv(a=0) = {:.5f}".format(integral_cv(1e-4)))  # deberia dar ~.5

print(integral_cv(np.array([1e4, 1e-4])))  # testing that it works for arrays

# Ahora hay que implementar la búsqueda del punto de interseccion entre la
# integral_cv y el valor 0.05


def func_zero(a):
    return integral_cv(a) - 0.05


a_star, result = newton(func_zero, 1, full_output=True)
print("El valor buscado es {:.3f}".format(a_star))
print("Numero de iteraciones: {}".format(result.iterations))

plt.figure(2, figsize=(6, 4))
plt.clf()

a_to_plot = np.linspace(0.5, 5, 100)
plt.plot(a_to_plot, integral_cv(a_to_plot))

plt.xlabel("$a$", fontsize=12)
plt.ylabel(r"$\int_0^{1/a} f(u) du $", fontsize=12)
plt.subplots_adjust(top=0.95, right=0.95, left=0.15, bottom=0.15)

plt.axhline(0.05, ls='--')
plt.axvline(a_star, ls='--', color='r', label="$a^*={:.2f}$".format(a_star))

plt.legend()
plt.savefig("solucion.pdf")
plt.show()


print("=====\n")
a_star, result = newton(func_zero, 0.5, full_output=True)
print("El valor buscado es {:.3f}".format(a_star))
print("Numero de iteraciones: {}".format(result.iterations))
