x_tiles = int(input('# of tiles horizontally: '))
y_tiles = int(input('# of tiles vertically: '))
tile_width = int(input('tile width: '))
tile_height = int(input('tile height: '))

from PIL import Image
import subprocess
import os

# build executable
subprocess.run(['cargo', 'build', '--release'])

handles = {}
for x in range(x_tiles):
    for y in range(y_tiles):
        if os.path.isfile(f'output/tile_{x}_{y}.png'):
            continue
        handle = subprocess.Popen(
            ['target/release/opencl-raytracing', f'tile_{x}_{y}.png', str(x * tile_width), str(y * tile_height), str(tile_width), str(tile_height)], 
            stdout=subprocess.DEVNULL
        )
        handles[handle.pid] = handle
        while len(handles) > 30:
            pid, _ = os.wait()
            del handles[pid]

[handle.wait() for handle in handles.values()]

merged = Image.new('RGB', (x_tiles * tile_width, y_tiles * tile_height))
for x in range(x_tiles):
    for y in range(y_tiles):
        tile = Image.open(f'output/tile_{x}_{y}.png')
        merged.paste(tile, (x * tile_width, y * tile_height))

merged.save('output/merged.png')