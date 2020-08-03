# -* - coding: UTF-8 -* -
from __future__ import with_statement
from __future__ import print_function
import os
import configparser
import random
import numpy as np 
import math


def main():
	cf = configparser.ConfigParser() 
	cf.read("test.cfg")
	layers = int(cf.get("info","layers"))
	conv = int(cf.get("info","conv"))
	fc = int(cf.get("info","fc"))
	operation_energy = float(cf.get("info","operation_energy"))
	mem_energy = float(cf.get("info","mem_energy"))
	memover_unstr = 0
	computationover_unstr = 0
	energyover_unstr = 0
	memover_str_perlay = 0
	memreadwrite_str_perlay = 0
	computationover_str_perlay = 0
	energyover_str_perlay = 0
	memover_str_perneu = 0
	memreadwrite_str_perneu = 0
	computationover_str_perneu = 0
	energyover_str_perneu = 0

	for i in range(0,layers):
		if (i<conv):
			layer_name = 'conv' + str(i+1)
			kernel = int(cf.get(layer_name,"kernel"))
			channel_out = int(cf.get(layer_name,"channel_out"))
			channel_in = int(cf.get(layer_name,"channel_in"))
			opw = int(cf.get(layer_name,"opw"))
			oph = int(cf.get(layer_name,"oph"))
			density_unstr = float(cf.get(layer_name,"density_unstr"))
			density_str_perlay = float(cf.get(layer_name,"density_str_perlay"))
			density_str_perneu = float(cf.get(layer_name,"density_str_perneu"))
			representation = int(cf.get(layer_name,"repre"))
			pt = float(cf.get(layer_name,"importance_percentage"))
			#memory overhead = output feature map size * input channel * kernel size * data representation
			memover_unstr = memover_unstr + opw*oph*channel_in*channel_out*kernel*kernel*representation
			memreadwrite_str_perlay = memreadwrite_str_perlay + opw*oph*channel_in*channel_out*kernel*kernel*representation
			memreadwrite_str_perneu = memreadwrite_str_perneu + opw*oph*channel_in*channel_out*kernel*kernel*representation*density_str_perneu

			if (opw*oph*channel_in*channel_out*kernel*kernel*representation > memover_str_perlay):
				memover_str_perlay = opw*oph*channel_in*channel_out*kernel*kernel*representation

			if (kernel*kernel*channel_in*representation) > memover_str_perneu:
				memover_str_perneu = kernel*kernel*channel_in*representation

			#computation overhead = sorting overhead + selection overhead
			useful_neuron_unstr = opw*oph*channel_out*density_unstr
			useful_neuron_str_perlay = opw*oph*channel_out*density_str_perlay
			useful_neuron_str_perneu = opw*oph*channel_out*density_str_perneu
			computationover_unstr = computationover_unstr + math.log(useful_neuron_unstr*kernel*kernel*channel_in,2) + useful_neuron_unstr*kernel*kernel*channel_in*pt
			computationover_str_perlay = computationover_str_perlay + math.log(opw*oph*channel_out,2) + math.log(useful_neuron_str_perlay*kernel*kernel*channel_in,2) + useful_neuron_str_perlay*kernel*kernel*channel_in*pt
			computationover_str_perneu = computationover_str_perneu + useful_neuron_str_perneu*kernel*kernel*channel_in*pt
			#print(layer_name)
		else:
			layer_name = 'fc' + str(i-conv+1)
			filterw = int(cf.get(layer_name,"filterw"))
			filterh = int(cf.get(layer_name,"filterh"))
			opw = int(cf.get(layer_name,"opw"))
			oph = int(cf.get(layer_name,"oph"))
			density_unstr = float(cf.get(layer_name,"density_unstr"))
			density_str_perlay = float(cf.get(layer_name,"density_str_perlay"))
			representation = float(cf.get(layer_name,"repre"))
			pt = float(cf.get(layer_name,"importance_percentage"))
			#print(layer_name)

			#memory overhead = output feature map size * input channel * kernel size * data representation			
			memover_unstr = memover_unstr + opw*oph*filterh*filterw*representation
			memreadwrite_str_perlay = memreadwrite_str_perlay + opw*oph*filterh*filterw*representation
			memreadwrite_str_perneu = memreadwrite_str_perneu + opw*oph*filterh*filterw*representation

			if (opw*oph*filterh*filterw*representation > memover_str_perlay):
				memover_str_perlay = opw*oph*filterh*filterw*representation

			if (filterh*filterw*representation) > memover_str_perneu:
				memover_str_perneu = filterh*filterw*representation

			#computation overhead 
			useful_neuron_unstr = opw*oph*density_unstr
			useful_neuron_str_perlay = opw*oph*density_str_perlay
			useful_neuron_str_perneu = opw*oph*density_str_perneu
			computationover_unstr = computationover_unstr + math.log(useful_neuron_unstr*filterw*filterh,2) + useful_neuron_unstr*filterw*filterh*pt
			computationover_str_perlay = computationover_str_perlay + math.log(opw*oph,2) + math.log(useful_neuron_str_perlay*filterw*filterh,2) + useful_neuron_str_perlay*filterw*filterh*pt
			computationover_str_perneu = computationover_str_perneu + useful_neuron_str_perneu*filterw*filterh*pt

	energyover_unstr = computationover_unstr * operation_energy + memover_unstr * mem_energy
	energyover_str_perlay = computationover_str_perlay * operation_energy + memreadwrite_str_perlay * mem_energy
	energyover_str_perneu = computationover_str_perneu * operation_energy + memreadwrite_str_perneu * mem_energy

	print("unstruct memory overhead",memover_unstr)
	print("unstruct computation overhead",computationover_unstr)
	print("unstuct energy overhead", energyover_unstr)
	print("struct per layer memory overhead", memover_str_perlay)
	print("struct per layer computation overhead", computationover_str_perlay)
	print("struct per layer energy overhead", energyover_str_perlay)
	print("struct per neuron memory overhead", memover_str_perneu)
	print("struct per neuron computation overhead", computationover_str_perneu)
	print("struct per neuron energy overhead", energyover_str_perneu)






if __name__ == '__main__':
	main()

