import generator
import clipmaker

print("Name for your clip:")
name = str(input())
print("Number of objects:")
objects = int(input())
print("Time length of your clip")
t = int(input())
print("Time to turn on light:")
t1 = int(input())
print("Time to turn off light:")
t2 = int(input())

generator.create_clip(20, objects, t, name, generator.velocities, generator.turn_rates, [t1], [t2])

clipmaker.make_clip(name)