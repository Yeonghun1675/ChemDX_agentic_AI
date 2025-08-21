from pathlib import Path

path = Path(__file__)

print (path)
new_path = (path).parent / "../database/data1.csv"

new_path = new_path.resolve()
print (new_path)