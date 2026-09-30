"""LEWIS.py -- common-molecule and custom Lewis structure app for TI-84 Evo.

Choose a common molecule or build a formula with the arrow keys. The app
draws the selected Lewis structure directly; CLEAR returns to the menu.
"""

import math
import ti_draw as d
import ti_system as tis
from CHEMCORE import *

LEFT = 24
UP = 25
RIGHT = 26
DOWN = 34
SECOND = 21
CLEAR = 45
ENTER = 105

WHITE = (248, 250, 249)
BLACK = (25, 40, 48)
NAVY = (24, 47, 56)
GREY = (102, 121, 128)
PALE = (224, 235, 232)
TEAL = (15, 112, 94)
RED = (178, 57, 57)
ELEMENT_CHOICES = ("H", "B", "C", "N", "O", "F", "Si", "P", "S",
                   "Cl", "Br", "I", "Xe")
PRESETS = (
    ("H2O", (("H", 2), ("O", 1)), 0),
    ("CO2", (("C", 1), ("O", 2)), 0),
    ("CH4", (("C", 1), ("H", 4)), 0),
    ("NH3", (("N", 1), ("H", 3)), 0),
    ("HCN", (("H", 1), ("C", 1), ("N", 1)), 0),
    ("BF3", (("B", 1), ("F", 3)), 0),
    ("SO2", (("S", 1), ("O", 2)), 0),
    ("NH4+", (("N", 1), ("H", 4)), 1),
    ("NO3-", (("N", 1), ("O", 3)), -1),
    ("CO3^2-", (("C", 1), ("O", 3)), -2),
    ("Custom formula...", None, 0),
)

BUFFERED = False
try:
    if hasattr(d, "use_buffer") and hasattr(d, "paint_buffer"):
        d.use_buffer()
        BUFFERED = True
except:
    BUFFERED = False


def color(rgb):
    d.set_color(rgb[0], rgb[1], rgb[2])


def present():
    if BUFFERED:
        d.paint_buffer()


def screen():
    color(WHITE)
    d.fill_rect(0, 0, 320, 210)


def header(label):
    color(NAVY)
    d.fill_rect(0, 0, 320, 26)
    color(TEAL)
    d.fill_rect(0, 24, 320, 2)
    color(WHITE)
    d.draw_text(5, 19, label[:31])


def footer(label):
    color(NAVY)
    d.fill_rect(0, 189, 320, 21)
    color(TEAL)
    d.fill_rect(0, 189, 320, 2)
    color(WHITE)
    d.draw_text(4, 206, label[:31])


def formula_text(slots, charge):
    out = ""
    for atom, count in slots:
        if count:
            out += atom
            if count > 1:
                out += str(count)
    if not out:
        out = "(empty)"
    if charge:
        if abs(charge) != 1:
            out += str(abs(charge))
        out += "+" if charge > 0 else "-"
    return out


def draw_menu(selected):
    screen()
    header("LEWIS STRUCTURES")
    color(BLACK)
    d.draw_text(8, 44, "Choose a molecule or ion:")
    first = max(0, min(selected - 6, len(PRESETS) - 7))
    for row in range(7):
        i = first + row
        if i >= len(PRESETS):
            break
        y = 67 + row * 17
        label = PRESETS[i][0][:25]
        label_x = 160 - len(label) * 5
        if i == selected:
            color(PALE)
            # Text sits higher than the TI rectangle's visual center, so lift
            # the selection band to center it on the selected label.
            d.fill_rect(6, y - 20, 306, 18)
            color(TEAL)
            d.draw_text(max(8, label_x - 17), y, ">")
        else:
            color(BLACK)
        d.draw_text(label_x, y, label)
    color(GREY)
    if first > 0:
        d.draw_text(292, 62, "^")
    if first + 7 < len(PRESETS):
        d.draw_text(292, 181, "v")
    footer("UP/DN PICK  ENTER OPEN  CLEAR")
    present()


def draw_editor(slots, focus, charge, count_mode, message):
    screen()
    header("CUSTOM FORMULA")
    color(BLACK)
    d.draw_text(8, 45, "Four element types; counts 0-6")
    for i in range(4):
        y = 72 + i * 23
        atom, count = slots[i]
        if i == focus:
            color(PALE)
            d.fill_rect(6, y - 14, 177, 20)
        color(TEAL if i == focus else BLACK)
        d.draw_text(13, y, ">" if i == focus else " ")
        d.draw_text(34, y, atom)
        d.draw_text(76, y, "x")
        d.draw_text(96, y, str(count))
        if i == focus:
            d.draw_text(133, y, "COUNT" if count_mode else "ATOM")

    if focus == 4:
        color(PALE)
        d.fill_rect(194, 121, 119, 31)
    color(BLACK)
    d.draw_text(202, 72, "Formula")
    color(TEAL)
    d.draw_text(202, 91, formula_text(slots, charge)[:11])
    color(BLACK)
    d.draw_text(202, 123, "Charge")
    color(TEAL)
    d.draw_text(202, 142, str(charge))
    color(GREY)
    d.draw_text(8, 174, message[:30])
    footer("2ND FIELD  UP/DN  ENTER DRAW")
    present()


def _charge_label(value):
    if value == 0:
        return ""
    if abs(value) == 1:
        return "+" if value > 0 else "-"
    return ("+" if value > 0 else "-") + str(abs(value))


def _pair_marks(x, y, angle, count, spacing=16, distance=14):
    """Draw lone-pair electron dots at a chosen distance from an atom."""
    ux, uy = math.cos(angle), math.sin(angle)
    px, py = -uy, ux
    color(BLACK)
    for pair in range(count):
        shift = (pair - (count - 1) / 2.0) * spacing
        mx = x + int(ux * distance + px * shift)
        my = y + int(uy * distance + py * shift)
        dx, dy = int(round(px * 4)), int(round(py * 4))
        d.fill_circle(mx - dx, my - dy, 2)
        d.fill_circle(mx + dx, my + dy, 2)


def _central_pair_angles(pair_count, ligand_count):
    if ligand_count == 2 and pair_count == 2:
        # Water's two lone pairs sit opposite its two bonds, below O in the
        # chosen sketch orientation, rather than crowded at O's left and right.
        return [math.pi / 2.0 - 0.58, math.pi / 2.0 + 0.58]
    if ligand_count == 2 and pair_count == 3:
        return [0.0, math.pi / 2.0, math.pi]
    if ligand_count == 4 and pair_count == 2:
        return [-math.pi / 2.0, math.pi / 2.0]
    if ligand_count == 3 and pair_count == 1:
        return [math.pi / 2.0]
    if ligand_count == 4 and pair_count == 1:
        return [math.pi / 2.0]
    return [-math.pi / 2.0 + 2.0 * math.pi * i / pair_count
            for i in range(pair_count)]


def _positions(ligand_count, center_lone_pairs, cx, cy):
    if ligand_count == 2:
        if center_lone_pairs:
            # A bent sketch (about 105 degrees) for a center with lone pairs.
            return [(-math.pi / 2.0 - 0.91), (-math.pi / 2.0 + 0.91)]
        return [-math.pi / 2.0, math.pi / 2.0]
    start = -math.pi / 2.0 if ligand_count == 3 else -math.pi / 4.0
    return [start + 2.0 * math.pi * i / ligand_count
            for i in range(ligand_count)]


def draw_structure(result, formula, resonance_index):
    screen()
    header("LEWIS: " + formula[:23])
    ligands = result["ligands"]
    center = result["center"]
    cx, cy = 160, 111
    radius = 58 if len(ligands) >= 5 else 62
    angles = _positions(len(ligands), result["lone_pairs"], cx, cy)
    positions = []
    for angle in angles:
        positions.append((cx + int(math.cos(angle) * radius),
                          cy + int(math.sin(angle) * radius), angle))

    forms = result["resonance"]
    orders = forms[resonance_index % len(forms)]
    terminal_lone = []
    for i, atom in enumerate(ligands):
        terminal_lone.append(0 if atom == "H" else 8 - 2 * orders[i])
    central_lone = result["electrons"] - 2 * sum(orders) - sum(terminal_lone)
    central_charge = LEWIS_VALENCE[center] - central_lone - sum(orders)
    charges = []
    for i, atom in enumerate(ligands):
        charges.append(LEWIS_VALENCE[atom] - terminal_lone[i] - orders[i])

    # Draw bonds first so labels and electron pairs stay visible.
    color(BLACK)
    for i, (x, y, angle) in enumerate(positions):
        order = orders[i]
        px, py = -math.sin(angle), math.cos(angle)
        offsets = [0] if order == 1 else ([-2, 2] if order == 2 else [-3, 0, 3])
        start = 13 if len(center) == 1 else 18
        for offset in offsets:
            d.draw_line(cx + int(math.cos(angle) * start) + int(px * offset),
                        cy + int(math.sin(angle) * start) + int(py * offset),
                        x - int(math.cos(angle) * 10) + int(px * offset),
                        y - int(math.sin(angle) * 10) + int(py * offset))

    for i, (x, y, angle) in enumerate(positions):
        atom = ligands[i]
        color(WHITE)
        d.fill_rect(x - len(atom) * 5 - 2, y - 11, len(atom) * 10 + 4, 21)
        color(TEAL)
        d.draw_text(x - len(atom) * 5, y + 5, atom)
        q = _charge_label(charges[i])
        if q:
            color(RED)
            d.draw_text(x + len(atom) * 5, y - 8, q[:3])
        pair_count = terminal_lone[i] // 2
        if pair_count:
            _pair_marks(x, y, angle, pair_count, distance=14)

    # A small white label clears the bond under the central atom symbol.
    color(WHITE)
    d.fill_rect(cx - len(center) * 5 - 3, cy - 11,
                len(center) * 10 + 6, 22)
    color(NAVY)
    d.draw_text(cx - len(center) * 5, cy + 5, center)
    if central_lone > 0:
        for angle in _central_pair_angles(central_lone // 2, len(ligands)):
            _pair_marks(cx, cy, angle, 1, distance=13)
    if central_charge:
        color(RED)
        d.draw_text(cx + len(center) * 5 + 2, cy - 9,
                    _charge_label(central_charge)[:3])

    color(GREY)
    if len(forms) > 1:
        d.draw_text(8, 45, "Resonance " + str(resonance_index + 1) + "/" + str(len(forms)))
    else:
        d.draw_text(8, 45, "Formal charges shown")
    footer("LEFT/RIGHT RES  CLEAR MENU")
    present()


def draw_error(message):
    screen()
    header("CANNOT DRAW FORMULA")
    color(RED)
    d.draw_text(8, 57, message[:30])
    color(BLACK)
    d.draw_text(8, 87, "Use one central atom and")
    d.draw_text(8, 105, "terminal atoms around it.")
    d.draw_text(8, 135, "Multi-center structures such")
    d.draw_text(8, 153, "as C2H6 are not supported yet.")
    footer("ENTER EDIT  CLEAR MENU")
    present()


def make_slots(atoms):
    slots = [["H", 0], ["C", 0], ["O", 0], ["N", 0]]
    for atom, count in atoms:
        found = False
        for slot in slots:
            if slot[0] == atom:
                slot[1] = count
                found = True
                break
        if not found:
            for slot in slots:
                if slot[1] == 0:
                    slot[0] = atom
                    slot[1] = count
                    found = True
                    break
        if not found:
            raise ValueError("Use at most four different elements.")
    return slots


def main():
    mode = "menu"
    selected = 0
    slots = [["H", 2], ["C", 0], ["O", 1], ["N", 0]]
    focus = 0
    charge = 0
    count_mode = False
    message = "Choose an element or count."
    result = None
    formula = ""
    resonance = 0
    error = ""

    while True:
        if mode == "menu":
            draw_menu(selected)
        elif mode == "edit":
            draw_editor(slots, focus, charge, count_mode, message)
        elif mode == "diagram":
            draw_structure(result, formula, resonance)
        else:
            draw_error(error)

        key = tis.wait_key()
        if mode == "menu":
            if key == CLEAR:
                break
            if key == UP:
                selected = (selected - 1) % len(PRESETS)
            elif key == DOWN:
                selected = (selected + 1) % len(PRESETS)
            elif key == ENTER:
                label, atoms, charge = PRESETS[selected]
                if atoms is None:
                    slots = [["H", 0], ["C", 0], ["O", 0], ["N", 0]]
                    focus = 0
                    charge = 0
                    count_mode = False
                    mode = "edit"
                else:
                    try:
                        result = build_lewis(list(atoms), charge)
                        formula = label
                        resonance = 0
                        mode = "diagram"
                    except Exception as exc:
                        error = str(exc)
                        mode = "error"
        elif mode == "edit":
            if key == CLEAR:
                mode = "menu"
            elif key == SECOND:
                if focus < 4:
                    count_mode = not count_mode
            elif key == UP:
                if count_mode and focus < 4:
                    slots[focus][1] = min(6, slots[focus][1] + 1)
                else:
                    focus = (focus - 1) % 5
            elif key == DOWN:
                if count_mode and focus < 4:
                    slots[focus][1] = max(0, slots[focus][1] - 1)
                else:
                    focus = (focus + 1) % 5
            elif key == LEFT:
                if focus == 4:
                    charge = max(-3, charge - 1)
                elif not count_mode:
                    atom = slots[focus][0]
                    slots[focus][0] = ELEMENT_CHOICES[(ELEMENT_CHOICES.index(atom) - 1) % len(ELEMENT_CHOICES)]
            elif key == RIGHT:
                if focus == 4:
                    charge = min(3, charge + 1)
                elif not count_mode:
                    atom = slots[focus][0]
                    slots[focus][0] = ELEMENT_CHOICES[(ELEMENT_CHOICES.index(atom) + 1) % len(ELEMENT_CHOICES)]
            elif key == ENTER:
                try:
                    atoms = [(s[0], s[1]) for s in slots if s[1] > 0]
                    result = build_lewis(atoms, charge)
                    formula = formula_text(slots, charge)
                    resonance = 0
                    mode = "diagram"
                except Exception as exc:
                    error = str(exc)
                    mode = "error"
        elif mode == "diagram":
            if key == CLEAR:
                mode = "menu"
            elif key == LEFT or key == RIGHT:
                resonance = (resonance + 1) % len(result["resonance"])
        elif mode == "error":
            if key == CLEAR:
                mode = "menu"
            elif key == ENTER:
                mode = "edit"

    screen()
    present()


main()
