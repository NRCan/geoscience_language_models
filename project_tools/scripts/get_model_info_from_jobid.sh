# Copyright (C) 2021 ServiceNow, Inc.

for JOBID in \
da274a8f-f80e-4849-afb6-3098acad3be7 \
aab4308a-7876-4aec-94f8-f2b5469fb857



do
  echo "JOB_ID:" $JOBID
  eai job info $JOBID | grep WANDB_PROJECT
  eai job logs $JOBID | head -37 | tail -1 # max_steps
  eai job logs $JOBID | head -33 | tail -1 # learning_rate
  eai job logs $JOBID | head -38 | tail -1 # warmup_steps
  eai job info $JOBID | grep EAI_EXPERIMENT_PARAMETERS_PATH
  print ""
  print ""
done
