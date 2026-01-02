"""
Management command to seed default compute providers.
Location: compute/management/commands/seed_providers.py
"""
from django.core.management.base import BaseCommand
from compute.models import ComputeProvider


class Command(BaseCommand):
    help = 'Seeds default compute providers with instance configurations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing providers',
        )

    def handle(self, *args, **options):
        overwrite = options['overwrite']
        
        providers_data = [
            {
                'name': 'Platform GPU Workers',
                'provider_type': 'platform-gpu',
                'description': 'High-performance GPU instances managed by the platform',
                'requires_user_credentials': False,
                'is_active': True,
                'system_config': {},
                'available_instances': [
                    {'name': 'gpu-small', 'vcpus': 4, 'memory_gb': 16, 'gpu_type': 'NVIDIA T4', 'gpu_count': 1, 'cost_per_hour': 0.50},
                    {'name': 'gpu-medium', 'vcpus': 8, 'memory_gb': 32, 'gpu_type': 'NVIDIA A10G', 'gpu_count': 1, 'cost_per_hour': 1.20},
                    {'name': 'gpu-large', 'vcpus': 16, 'memory_gb': 64, 'gpu_type': 'NVIDIA A100', 'gpu_count': 1, 'cost_per_hour': 3.50},
                    {'name': 'gpu-xlarge', 'vcpus': 32, 'memory_gb': 128, 'gpu_type': 'NVIDIA A100', 'gpu_count': 4, 'cost_per_hour': 12.00},
                ],
            },
            {
                'name': 'Platform CPU Workers',
                'provider_type': 'platform-cpu',
                'description': 'Cost-effective CPU instances for lighter workloads',
                'requires_user_credentials': False,
                'is_active': True,
                'system_config': {},
                'available_instances': [
                    {'name': 'cpu-small', 'vcpus': 2, 'memory_gb': 4, 'gpu_count': 0, 'cost_per_hour': 0.05},
                    {'name': 'cpu-medium', 'vcpus': 4, 'memory_gb': 16, 'gpu_count': 0, 'cost_per_hour': 0.12},
                    {'name': 'cpu-large', 'vcpus': 8, 'memory_gb': 32, 'gpu_count': 0, 'cost_per_hour': 0.25},
                ],
            },
            {
                'name': 'AWS SageMaker',
                'provider_type': 'aws-sagemaker',
                'description': 'Fully managed ML service with auto-scaling capabilities',
                'requires_user_credentials': True,
                'is_active': True,
                'system_config': {
                    'region': 'us-east-1',
                    'use_spot_instances': False
                },
                'available_instances': [
                    {'name': 'ml.g4dn.xlarge', 'vcpus': 4, 'memory_gb': 16, 'gpu_type': 'NVIDIA T4', 'gpu_count': 1, 'cost_per_hour': 0.73},
                    {'name': 'ml.g5.2xlarge', 'vcpus': 8, 'memory_gb': 32, 'gpu_type': 'NVIDIA A10G', 'gpu_count': 1, 'cost_per_hour': 1.52},
                    {'name': 'ml.p4d.24xlarge', 'vcpus': 96, 'memory_gb': 1152, 'gpu_type': 'NVIDIA A100', 'gpu_count': 8, 'cost_per_hour': 37.69},
                ],
            },
            {
                'name': 'GCP Vertex AI',
                'provider_type': 'gcp-vertex',
                'description': 'Google Cloud AI platform with TPU support',
                'requires_user_credentials': True,
                'is_active': True,
                'system_config': {
                    'region': 'us-central1'
                },
                'available_instances': [
                    {'name': 'n1-highmem-4-t4', 'vcpus': 4, 'memory_gb': 26, 'gpu_type': 'NVIDIA T4', 'gpu_count': 1, 'cost_per_hour': 0.65},
                    {'name': 'a2-highgpu-1g', 'vcpus': 12, 'memory_gb': 85, 'gpu_type': 'NVIDIA A100', 'gpu_count': 1, 'cost_per_hour': 4.01},
                ],
            },
            {
                'name': 'Azure ML',
                'provider_type': 'azure-ml',
                'description': 'Microsoft Azure machine learning compute',
                'requires_user_credentials': True,
                'is_active': True,
                'system_config': {},
                'available_instances': [
                    {'name': 'Standard_NC4as_T4_v3', 'vcpus': 4, 'memory_gb': 28, 'gpu_type': 'NVIDIA T4', 'gpu_count': 1, 'cost_per_hour': 0.53},
                    {'name': 'Standard_NC24ads_A100_v4', 'vcpus': 24, 'memory_gb': 220, 'gpu_type': 'NVIDIA A100', 'gpu_count': 1, 'cost_per_hour': 3.67},
                ],
            },
            {
                'name': 'RunPod',
                'provider_type': 'runpod',
                'description': 'On-demand GPU cloud with competitive pricing',
                'requires_user_credentials': True,
                'is_active': True,
                'system_config': {},
                'available_instances': [
                    {'name': 'RTX 3090', 'vcpus': 8, 'memory_gb': 24, 'gpu_type': 'NVIDIA RTX 3090', 'gpu_count': 1, 'cost_per_hour': 0.44},
                    {'name': 'RTX 4090', 'vcpus': 16, 'memory_gb': 48, 'gpu_type': 'NVIDIA RTX 4090', 'gpu_count': 1, 'cost_per_hour': 0.69},
                    {'name': 'A100 80GB', 'vcpus': 32, 'memory_gb': 128, 'gpu_type': 'NVIDIA A100 80GB', 'gpu_count': 1, 'cost_per_hour': 1.99},
                ],
            },
            {
                'name': 'On-Premise Agent',
                'provider_type': 'on-premise-agent',
                'description': 'Connect your own GPU infrastructure',
                'requires_user_credentials': False,
                'is_active': True,
                'system_config': {},
                'available_instances': [
                    {'name': 'custom', 'vcpus': 0, 'memory_gb': 0, 'gpu_count': 0, 'cost_per_hour': 0.0},
                ],
            },
        ]
        
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for provider_data in providers_data:
            provider_type = provider_data['provider_type']
            
            # Check if exists
            existing = ComputeProvider.objects.filter(provider_type=provider_type).first()
            
            if existing:
                if overwrite:
                    # Update existing
                    for key, value in provider_data.items():
                        setattr(existing, key, value)
                    existing.save()
                    updated_count += 1
                    self.stdout.write(
                        self.style.WARNING(f'Updated: {provider_data["name"]}')
                    )
                else:
                    skipped_count += 1
                    self.stdout.write(
                        self.style.NOTICE(f'Skipped (already exists): {provider_data["name"]}')
                    )
            else:
                # Create new
                ComputeProvider.objects.create(**provider_data)
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created: {provider_data["name"]}')
                )
        
        # Summary
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.SUCCESS(f'Created: {created_count}'))
        self.stdout.write(self.style.WARNING(f'Updated: {updated_count}'))
        self.stdout.write(self.style.NOTICE(f'Skipped: {skipped_count}'))
        self.stdout.write(self.style.SUCCESS(f'Total providers: {ComputeProvider.objects.count()}'))
        self.stdout.write(self.style.SUCCESS('=' * 50))
        
        if skipped_count > 0 and not overwrite:
            self.stdout.write('')
            self.stdout.write(
                self.style.NOTICE('Tip: Use --overwrite to update existing providers')
            )