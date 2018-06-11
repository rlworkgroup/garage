#!/usr/bin/env bash
flake8 --import-order-style=google --application-import-names=sandbox,rllab,examples,contrib --select=I100,I101,I201,I202
