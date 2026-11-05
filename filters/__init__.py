from .aranhaverso import filtro_aranhaverso
from .bordas import filtro_bordas
from .cartoon import filtro_cartoon
from .ciber_neon import filtro_ciber_neon
from .esboco import filtro_esboco
from .gengar import filtro_gengar
from .glitch import filtro_glitch
from .homem_de_ferro import filtro_homem_de_ferro
from .minecraft import filtro_minecraft
from .minecraft_diamante import filtro_minecraft_diamante
from .minecraft_ender import filtro_minecraft_ender
from .minecraft_nether import filtro_minecraft_nether
from .pixel_8bit import filtro_pixel_8bit
from .pixel_jogo import filtro_pixel_jogo
from .termico import filtro_termico
from .vhs import filtro_vhs

FILTROS = [
    filtro_minecraft, filtro_minecraft_diamante,
    filtro_minecraft_nether, filtro_minecraft_ender,
    filtro_pixel_jogo, filtro_pixel_8bit,
    filtro_homem_de_ferro, filtro_aranhaverso, filtro_gengar, filtro_vhs,
    filtro_termico, filtro_esboco, filtro_glitch, filtro_bordas,
    filtro_cartoon, filtro_ciber_neon,
]
