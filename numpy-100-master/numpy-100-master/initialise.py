##########################################################
# initialise.py
##########################################################

from IPython.display import display, Markdown, HTML, Javascript
import ipywidgets as widgets
import numpy as np
import generators as ge


# =========================
# Funciones "clásicas"
# =========================
def question(n):
    """
    Muestra la pregunta en formato Markdown.
    """
    display(Markdown(f"### ❓ Question {n}\n> {ge.QHA[f'q{n}']}"))


def hint(n):
    """
    Muestra la pista en formato Markdown.
    """
    display(Markdown(f"💡 **Hint:** `{ge.QHA[f'h{n}']}`"))


def answer(n):
    """
    Muestra la respuesta en formato Markdown (bloque de código).
    """
    display(Markdown(f"✅ **Answer:**\n```python\n{ge.QHA[f'a{n}']}\n```"))


def pick():
    """
    Selecciona una pregunta aleatoria y la muestra con el estilo clásico.
    """
    n = np.random.randint(1, 101)
    question(n)


# =========================
# Funciones Interactivas
# =========================
def spawn_code_cell():
    """
    Inserta dinámicamente una celda de código debajo de la actual.
    Funciona únicamente en el Jupyter Notebook clásico.
    """
    display(
        Javascript("""
    if (typeof Jupyter !== "undefined" && Jupyter.notebook) {
        var new_cell = Jupyter.notebook.insert_cell_below('code');
        new_cell.set_text("# Escribe tu respuesta aquí...");
        Jupyter.notebook.select_next();
    } else {
        console.warn("No se detecta el entorno Jupyter Notebook clásico. No se puede crear la celda.");
    }
    """)
    )


def question_interactive(n):
    """
    Muestra la pregunta en Markdown con fondo oscuro,
    y las pistas/respuestas en un Accordion colapsable.
    Al final, inserta una celda de código en el Notebook clásico.
    """

    import ipywidgets as widgets
    from IPython.display import display, HTML, Markdown
    import generators as ge

    # --- CSS personalizado para el Accordion oscuro ---
    custom_css = """
    <style>
    /* Ajusta el fondo y el texto en los encabezados y contenido del Accordion */
    .dark-accordion .accordion-header,
    .dark-accordion .accordion-content,
    .dark-accordion .p-Accordion-child {
        background-color: #2c313a !important;
        color: #fff !important;
        border: none !important;
    }
    /* Hover sobre el encabezado del Accordion */
    .dark-accordion .accordion-header:hover {
        background-color: #222 !important;
    }
    </style>
    """
    display(HTML(custom_css))

    # --- Mostramos la pregunta en Markdown (permite LaTeX con $...$) ---
    display(Markdown(f"# ❓ Pregunta {n}\n\n{ge.QHA[f'q{n}']}"))

    # --- Creamos los objetos Output para la pista y la solución ---
    out_hint = widgets.Output()
    with out_hint:
        display(Markdown(f"{ge.QHA[f'h{n}']}"))

    out_answer = widgets.Output()
    with out_answer:
        display(Markdown(f"```python\n{ge.QHA[f'a{n}']}\n```"))

    # --- Acordeón oscuro con 2 secciones: Pista y Solución ---
    accordion = widgets.Accordion(children=[out_hint, out_answer], selected_index=None)
    accordion.set_title(0, "Pista")
    accordion.set_title(1, "Solución")

    # Añadimos la clase "dark-accordion" para aplicar el estilo CSS definido arriba
    accordion.add_class("dark-accordion")

    # --- Mostramos el Accordion ---
    display(accordion)

    # --- Insertamos una celda de Python debajo de la actual (Notebook clásico) ---
    spawn_code_cell()


def pick_interactive():
    """
    Selecciona una pregunta aleatoria y la muestra en formato interactivo.
    """
    n = np.random.randint(1, 4)
    question_interactive(n)
